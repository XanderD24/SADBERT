"""
sadbert/core.py
───────────────
Core implementation of the SADBERT stereotype-content classifier.

Architecture
────────────
The pipeline runs in three stages:

  Stage 1 — Master model (XanderD24/SADBERT_master_model)
      Multi-label DistilBERT that generates per-class probabilities.
      Per-class thresholds from Youden's J statistic (ROC_dict.pkl)
      determine which categories are *candidate* predictions.

  Stage 2 — Classifier heads (XanderD24/SADBERT_{cat}_classifier)
      One binary DistilBERT per SADCAT category. Acts as a veto gate:
      a candidate category is kept only when the dedicated head also
      predicts positive. Returns the head's probability for that category.

  Stage 3 — Sentiment models (XanderD24/SADBERT_{cat}_sentiment)
      One 3-class DistilBERT per *major* category (those with a valence
      direction). Predicts negative / neutral / positive and returns both
      the class label and the human-readable interpretation.
"""

from __future__ import annotations

import math
import pickle
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Union, cast

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, BatchEncoding, DistilBertForSequenceClassification, PreTrainedTokenizerBase

# ── Bundled data directory ────────────────────────────────────────────────────
_DATA_DIR = Path(__file__).parent / "data"

# ── HuggingFace namespace ─────────────────────────────────────────────────────
_HF_USER = "XanderD24"

# ── Category lists ────────────────────────────────────────────────────────────
#   Major categories have both a classifier head AND a sentiment model.
#   Minor categories have only a classifier head.

MAJOR_CATS: List[str] = [
    "Warmth", "Competence", "Sociability", "Morality", "Ability",
    "Assertiveness", "Status", "Beliefs", "health", "deviance",
    "beauty", "Politics", "Religion",
]

MINOR_CATS: List[str] = [
    "emotions", "Geography", "Appearance", "occupation", "socialgroups",
    "inhabitant", "country", "relative", "insults", "stem", "humanities",
    "art", "Lacksknowledge", "fortune", "clothing", "bodpart", "bodprop",
    "skin", "bodcov", "beliefsother", "Other_large", "Other",
]

ALL_CATS: List[str] = MAJOR_CATS + MINOR_CATS

# Sentiment model label → direction value (from valence fine-tuning)
#   0 → negative (-1), 1 → neutral (0), 2 → positive (+1)
_LABEL_TO_DIR: Dict[int, int] = {0: -1, 1: 0, 2: 1}

# ── Output column names ───────────────────────────────────────────────────────
_COLS = ["category", "probability", "valence", "valence probability", "interpretation"]


# ─────────────────────────────────────────────────────────────────────────────
# SADBERT class
# ─────────────────────────────────────────────────────────────────────────────

class SADBERT:
    """
    Stereotype-content classifier built on a DistilBERT ensemble.

    Parameters
    ----------
    device : str | None
        Torch device string (``"cuda"``, ``"mps"``, ``"cpu"``).
        Auto-detected when ``None`` (default).
    batch_size : int
        Number of texts processed per forward pass.  Default 32.
    load_models : bool
        If ``True`` (default), all HuggingFace models are downloaded and
        loaded during ``__init__``.  Set to ``False`` to defer loading
        (models will be fetched on the first call to
        :meth:`get_stereotype_content`).

    Examples
    --------
    >>> import sadbert
    >>> results = sadbert.get_stereotype_content("She is a warm and caring nurse.")
    >>> print(results)
    """

    def __init__(
        self,
        device: Optional[str] = None,
        batch_size: int = 32,
        load_models: bool = True,
    ) -> None:
        # ── Device ────────────────────────────────────────────────────────────
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = torch.device(device)
        self.batch_size = batch_size

        # Internal state — populated by _load_models()
        self._loaded = False
        self.tokenizer: Optional[PreTrainedTokenizerBase] = None
        self.master_model: Optional[DistilBertForSequenceClassification] = None
        self.classifier_heads: Dict[str, DistilBertForSequenceClassification] = {}
        self.sentiment_models: Dict[str, DistilBertForSequenceClassification] = {}
        self._valid_idtolabel: Dict[int, str] = {}
        self._roc_thresholds: Dict[str, float] = {}
        self._interpretation_dict: Dict[str, Dict[int, str]] = {}

        if load_models:
            self._load_models()
    
    # ── Model + data loading ──────────────────────────────────────────────────

    def _load_model(self, repo_id: str) -> DistilBertForSequenceClassification:
        """
        Load a DistilBertForSequenceClassification from *repo_id*, move it to
        the configured device, and set it to eval mode.

        ``cast`` is used to preserve the concrete subtype through
        ``from_pretrained``, which is typed to return the base
        ``PreTrainedModel`` class.
        """
        model = cast(
            DistilBertForSequenceClassification,
            DistilBertForSequenceClassification.from_pretrained(repo_id),
        )
        model.to(self.device)  # type: ignore[call-arg]
        model.eval()
        return model

    def _load_models(self) -> None:
        """Download (if not cached) and load all models and static data."""
        if self._loaded:
            return

        print("Loading SADBERT models from HuggingFace Hub …")
        print(f"  Device: {self.device}")

        # ── Static data files ──────────────────────────────────────────────
        self._valid_idtolabel = self._load_idtolabel()
        self._roc_thresholds  = self._load_roc_dict()
        self._interpretation_dict = self._load_interpretation_dict()

        # ── Shared tokeniser (from master model) ──────────────────────────
        master_repo = f"{_HF_USER}/SADBERT_master_model"
        print(f"  Loading tokeniser from {master_repo} …")
        self.tokenizer = AutoTokenizer.from_pretrained(master_repo)

        # ── Master model ──────────────────────────────────────────────────
        print("  Loading master model …")
        self.master_model = self._load_model(master_repo)

        # ── Classifier heads (one per category) ───────────────────────────
        # Repo format: XanderD24/SADBERT_{category}_classifier
        # Dictionary key: lowercase category name
        print(f"  Loading {len(ALL_CATS)} classifier heads …")
        self.classifier_heads = {
            cat.lower(): self._load_model(f"{_HF_USER}/SADBERT_{cat}_classifier")
            for cat in ALL_CATS
        }

        # ── Sentiment models (major categories only) ───────────────────────
        # Repo format: XanderD24/SADBERT_{category}_sentiment
        # Dictionary key: lowercase category name
        print(f"  Loading {len(MAJOR_CATS)} sentiment models …")
        self.sentiment_models = {
            cat.lower(): self._load_model(f"{_HF_USER}/SADBERT_{cat}_sentiment")
            for cat in MAJOR_CATS
        }

        self._loaded = True
        print("All models loaded ✓")

    @staticmethod
    def _load_idtolabel() -> Dict[int, str]:
        """Load and filter idtolabel from bundled label_mappings.pkl."""
        with open(_DATA_DIR / "label_mappings.pkl", "rb") as f:
            mappings = pickle.load(f)
        raw = mappings["idtolabel"]
        return {
            idx: str(cat) for idx, cat in raw.items()
            if cat is not None
            and not (isinstance(cat, float) and math.isnan(cat))
            and str(cat) != "nan"
        }

    @staticmethod
    def _load_roc_dict() -> Dict[str, float]:
        """Load per-category probability thresholds from bundled ROC_dict.pkl."""
        roc_path = _DATA_DIR / "ROC_dict.pkl"
        if not roc_path.exists():
            raise FileNotFoundError(
                "ROC_dict.pkl not found in the package data directory.\n"
                f"Expected path: {roc_path}\n"
                "Please copy ROC_dict.pkl into sadbert/data/ before building "
                "the package. See README.md for instructions."
            )
        with open(roc_path, "rb") as f:
            return pickle.load(f)

    @staticmethod
    def _load_interpretation_dict() -> Dict[str, Dict[int, str]]:
        """Load valence-interpretation strings from bundled interpretation_dict.pkl."""
        with open(_DATA_DIR / "interpretation_dict.pkl", "rb") as f:
            return pickle.load(f)

    # ── Internal inference helpers ────────────────────────────────────────────

    def _ensure_loaded(self) -> None:
        if not self._loaded:
            self._load_models()

    def _tokenize(self, texts: List[str]) -> BatchEncoding:
        assert self.tokenizer is not None, (
            "Tokenizer is not loaded. Ensure _load_models() has been called."
        )
        return self.tokenizer(
            texts,
            return_tensors="pt",
            truncation=True,
            padding=True,
        )

    def _batch_forward(
        self,
        model: DistilBertForSequenceClassification,
        texts: List[str],
    ) -> torch.Tensor:
        """
        Run *model* on *texts* in batches.

        Returns
        -------
        torch.Tensor of shape ``(len(texts), num_labels)``
            Softmax-normalised probabilities on CPU.
        """
        all_probs: List[torch.Tensor] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            inputs = self._tokenize(batch)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                logits = model(**inputs).logits
            all_probs.append(torch.softmax(logits, dim=-1).cpu())
        return torch.cat(all_probs, dim=0)  # (N, num_labels)

    # ── Main public API ───────────────────────────────────────────────────────

    def get_stereotype_content(
        self,
        text: Union[str, List[str]],
        stacked: bool = True,
    ) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
        """
        Classify stereotype content dimensions and valences for one or more texts.

        Parameters
        ----------
        text : str | list[str]
            A single phrase/word or a list of phrases/words to analyse.
        stacked : bool
            - ``True`` (default): return a single DataFrame with all results
              plus a ``"text"`` column identifying the source text.
            - ``False``: return a ``dict[str, DataFrame]`` mapping each input
              text to its own results table.  For a single string input,
              returns the DataFrame directly.

        Returns
        -------
        pd.DataFrame
            When *stacked* is ``True`` or input is a single string with
            *stacked* ``False``.
        dict[str, pd.DataFrame]
            When *stacked* is ``False`` and input is a list.

        Notes
        -----
        The pipeline runs in three stages:

        1. **Master model** — softmax probabilities compared against per-class
           Youden-J thresholds to produce candidate categories.
        2. **Classifier heads** — binary veto gate; only categories confirmed
           by the dedicated head survive. The head's positive-class probability
           becomes the ``"probability"`` column.
        3. **Sentiment models** — for major categories only; predicts valence
           (−1 / 0 / +1) and maps to an interpretation string.
        """
        self._ensure_loaded()
        assert self.master_model is not None, (
            "Master model is not loaded. Ensure _load_models() completed successfully."
        )
        assert self.tokenizer is not None, (
            "Tokenizer is not loaded. Ensure _load_models() completed successfully."
        )

        # ── Normalise input ────────────────────────────────────────────────
        single = isinstance(text, str)
        texts: List[str] = [text] if single else list(text)
        n = len(texts)

        if n == 0:
            empty = pd.DataFrame(columns=_COLS)
            return {} if (not stacked and not single) else empty

        # ══════════════════════════════════════════════════════════════════
        # STAGE 1 — Master model: collect candidate categories per text
        # ══════════════════════════════════════════════════════════════════
        master_probs = self._batch_forward(self.master_model, texts)
        # master_probs: (n, num_master_classes)

        # all_candidates[text_idx] = list of category names that cleared threshold
        all_candidates: Dict[int, List[str]] = {}
        for i in range(n):
            candidates = []
            for class_idx, category in self._valid_idtolabel.items():
                threshold = self._roc_thresholds.get(category)
                if threshold is None:
                    continue
                if master_probs[i, class_idx].item() > threshold:
                    candidates.append(category)
            all_candidates[i] = candidates

        # ══════════════════════════════════════════════════════════════════
        # STAGE 2 — Classifier heads: veto gate, collect probabilities
        # Group text indices by candidate category for batched inference.
        # ══════════════════════════════════════════════════════════════════
        cat_to_text_idxs: Dict[str, List[int]] = defaultdict(list)
        for idx, cats in all_candidates.items():
            for cat in cats:
                cat_to_text_idxs[cat].append(idx)

        # confirmed[text_idx] = list of (category, classifier_prob)
        confirmed: Dict[int, List[tuple]] = defaultdict(list)

        for category, text_idxs in cat_to_text_idxs.items():
            key = category.lower()
            candidate_texts = [texts[i] for i in text_idxs]

            if key not in self.classifier_heads:
                # No head available — fall back to master prediction alone
                warnings.warn(
                    f"No classifier head found for '{category}'. "
                    "Keeping master model prediction.",
                    stacklevel=3,
                )
                for i in text_idxs:
                    confirmed[i].append((category, None))
                continue

            head = self.classifier_heads[key]
            head_probs = self._batch_forward(head, candidate_texts)
            # Column 1 = probability of positive class (belongs to category)
            pos_probs = head_probs[:, 1].tolist()

            for i, prob in zip(text_idxs, pos_probs):
                if prob > 0.5:                # head predicts positive
                    confirmed[i].append((category, prob))

        # ══════════════════════════════════════════════════════════════════
        # STAGE 3 — Sentiment models: valence for major categories
        # Again batched per category for efficiency.
        # ══════════════════════════════════════════════════════════════════
        # Collect (text_idx) per major category that survived Stage 2
        major_cat_idxs: Dict[str, List[int]] = defaultdict(list)
        major_cats_set = set(MAJOR_CATS)

        for idx, pairs in confirmed.items():
            for category, _ in pairs:
                if category in major_cats_set:
                    major_cat_idxs[category].append(idx)

        # sentiment_preds[text_idx][category] = (valence_int, valence_prob, interpretation)
        sentiment_preds: Dict[int, Dict[str, tuple]] = defaultdict(dict)

        for category, text_idxs in major_cat_idxs.items():
            key = category.lower()
            if key not in self.sentiment_models:
                warnings.warn(
                    f"No sentiment model found for major category '{category}'.",
                    stacklevel=3,
                )
                continue

            sent_model = self.sentiment_models[key]
            sent_texts = [texts[i] for i in text_idxs]
            sent_probs = self._batch_forward(sent_model, sent_texts)
            # sent_probs: (len(text_idxs), 3)   classes: 0=negative, 1=neutral, 2=positive

            for i, probs_row in zip(text_idxs, sent_probs):
                valence_class = int(probs_row.argmax().item())
                valence_prob  = float(probs_row[valence_class].item())
                valence_dir   = _LABEL_TO_DIR[valence_class]
                interpretation = self._interpretation_dict.get(category, {}).get(
                    valence_class, "Unknown"
                )
                sentiment_preds[i][category] = (valence_dir, valence_prob, interpretation)

        # ══════════════════════════════════════════════════════════════════
        # Assemble per-text DataFrames
        # ══════════════════════════════════════════════════════════════════
        results: Dict[str, pd.DataFrame] = {}

        for idx, input_text in enumerate(texts):
            cat_prob_pairs = confirmed.get(idx, [])

            if not cat_prob_pairs:
                # No category survived the pipeline
                df = pd.DataFrame(
                    [["None", None, "None", "None", "None"]],
                    columns=_COLS,
                )
            else:
                rows = []
                for category, cls_prob in cat_prob_pairs:
                    if category in major_cats_set and idx in sentiment_preds \
                            and category in sentiment_preds[idx]:
                        valence, val_prob, interpretation = sentiment_preds[idx][category]
                    else:
                        valence       = "None"
                        val_prob      = "None"
                        interpretation = "None"

                    rows.append({
                        "category":          category,
                        "probability":       cls_prob,
                        "valence":           valence,
                        "valence probability": val_prob,
                        "interpretation":    interpretation,
                    })
                df = pd.DataFrame(rows, columns=_COLS)

            results[input_text] = df

        # ── Return ─────────────────────────────────────────────────────────
        if not stacked:
            if single:
                return results[texts[0]]
            return dict(results)

        # stacked=True — concatenate all DataFrames, prepend a "text" column
        frames = []
        for input_text, df in results.items():
            df = df.copy()
            df.insert(0, "text", input_text)
            frames.append(df)

        if frames:
            return pd.concat(frames, ignore_index=True)

        return pd.DataFrame(
            columns=["text"] + _COLS
        )


# ─────────────────────────────────────────────────────────────────────────────
# Module-level convenience API
# ─────────────────────────────────────────────────────────────────────────────

_default_instance: Optional[SADBERT] = None


def _get_default_instance() -> SADBERT:
    """Return (and lazily create) the shared module-level SADBERT instance."""
    global _default_instance
    if _default_instance is None:
        _default_instance = SADBERT()
    return _default_instance


def get_stereotype_content(
    text: Union[str, List[str]],
    stacked: bool = True,
) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """
    Classify stereotype content dimensions and valences for one or more texts.

    This is a module-level convenience wrapper around :class:`SADBERT`.
    The underlying model instance is created lazily on the first call and
    reused for all subsequent calls.

    Parameters
    ----------
    text : str | list[str]
        A single phrase/word or a list of phrases/words to analyse.
    stacked : bool
        - ``True`` (default): return a single DataFrame with all results
          plus a ``"text"`` column.
        - ``False``: return a ``dict[str, DataFrame]``.  For a single string
          input, returns the DataFrame directly.

    Returns
    -------
    pd.DataFrame | dict[str, pd.DataFrame]
        See :meth:`SADBERT.get_stereotype_content` for full description.

    Examples
    --------
    Single word
    ~~~~~~~~~~~
    >>> import sadbert
    >>> sadbert.get_stereotype_content("honest")
       category  probability  valence  valence probability interpretation
    0    Warmth        0.912      1.0              0.876           Warm
    1   Morality        0.843      1.0              0.791          Moral

    Multiple words, unstacked
    ~~~~~~~~~~~~~~~~~~~~~~~~~
    >>> results = sadbert.get_stereotype_content(
    ...     ["honest", "lazy"],
    ...     stacked=False,
    ... )
    >>> results["lazy"]
       category  probability  valence  valence probability  interpretation
    0  Competence        0.791     -1.0              0.883     Incompetent

    Stacked (default)
    ~~~~~~~~~~~~~~~~~
    >>> sadbert.get_stereotype_content(["honest", "lazy"])
           text    category  probability  valence  valence probability interpretation
    0    honest      Warmth        0.912      1.0              0.876           Warm
    ...
    """
    return _get_default_instance().get_stereotype_content(text, stacked)
print("sadbert.core module loaded ✓")