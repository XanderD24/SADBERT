# SADBERT

**Stereotype-content Analysis with DistilBERT** — a Python package for identifying and characterising stereotype-relevant dimensions in natural language text, based on the SADCAT (Stereotype-Associated Dictionary of Content and Affect with Traits) framework.

## What it does

Given a word or phrase, SADBERT predicts:

| Output column | Description |
|---|---|
| `category` | Stereotype-content dimension (e.g. *Warmth*, *Competence*) |
| `probability` | Confidence of the category classifier |
| `valence` | Direction within the category: `1` = positive, `0` = neutral, `−1` = negative |
| `valence probability` | Confidence of the valence prediction |
| `interpretation` | Human-readable label (e.g. *"Warm"*, *"Incompetent"*, *"Moral"*) |

### Categories detected

**Major** (with valence): Warmth · Competence · Sociability · Morality · Ability · Assertiveness · Status · Beliefs · health · deviance · beauty · Politics · Religion

**Minor** (category only, no valence): emotions · Geography · Appearance · occupation · socialgroups · inhabitant · country · relative · insults · stem · humanities · art · Lacksknowledge · fortune · clothing · bodpart · bodprop · skin · bodcov · beliefsother · Other\_large · Other

---

## Installation

```bash
pip install sadbert
```

> **Note:** On first use, SADBERT automatically downloads ~2 GB of model weights from the HuggingFace Hub. These are cached locally in `~/.cache/huggingface/` and do not need to be re-downloaded on subsequent runs.

### GPU / Apple Silicon

SADBERT auto-detects CUDA and Apple MPS. To use a specific device, instantiate `SADBERT` directly:

```python
from sadbert import SADBERT
model = SADBERT(device="cuda")   # or "mps", "cpu"
```

---

## Quick Start

```python
import sadbert

# Single word — returns a DataFrame
df = sadbert.get_stereotype_content("honest")
print(df)

# Multiple words — stacked into one DataFrame (default)
df = sadbert.get_stereotype_content(["honest", "lazy", "senator"])
print(df)

# Multiple words — one DataFrame per word
results = sadbert.get_stereotype_content(["honest", "lazy"], stacked=False)
print(results["honest"])
print(results["lazy"])
```

### Example output

```
>>> sadbert.get_stereotype_content("honest")

   category  probability  valence  valence probability interpretation
0    Warmth        0.912      1.0               0.876           Warm
1  Morality        0.843      1.0               0.791          Moral
```

---

## API reference

### `sadbert.get_stereotype_content(text, stacked=True)`

Module-level convenience function. Uses a shared, lazily-initialised `SADBERT` instance.

| Parameter | Type | Description |
|---|---|---|
| `text` | `str` or `list[str]` | Word(s) or phrase(s) to classify |
| `stacked` | `bool` | `True` (default): return one combined DataFrame with a `"text"` column. `False`: return a `dict[str, DataFrame]`. For single string input with `stacked=False`, returns the DataFrame directly. |

---

### `sadbert.predict_individual_types(text, content_type=None, head_type=None, manual_specification=None, stacked=True)`

Run specific model heads against specific categories, without going through the full three-stage pipeline. Useful when you already know which categories you want to probe, or when you want classifier and sentiment scores independently.

| Parameter | Type | Description |
|---|---|---|
| `text` | `str` or `list[str]` | Word(s) or phrase(s) to classify |
| `content_type` | `str`, `list[str]`, or `None` | Category name(s) to analyse (e.g. `"Warmth"`, `["Warmth", "Competence"]`). Case-insensitive. |
| `head_type` | `str`, `list[str]`, or `None` | Which head(s) to run: `"classifier"`, `"sentiment"`, or both. Sentiment heads are silently skipped for minor categories (which have no valence model). |
| `manual_specification` | `list[tuple[str, str]]` or `None` | Explicit `(category, head_type)` pairs, e.g. `[("morality", "classifier"), ("warmth", "sentiment")]`. Can be combined with `content_type` / `head_type`. |
| `stacked` | `bool` | `True` (default): one combined DataFrame. `False`: `dict[str, DataFrame]` keyed by input text. |

At least one of `content_type`, `head_type`, or `manual_specification` must be provided.

**Output columns**

| Column | Description |
|---|---|
| `text` | Source text |
| `category` | Stereotype-content dimension |
| `model type` | `"classifier"` or `"sentiment"` |
| `probability` | For classifiers: probability of belonging to the category. For sentiment: full `[neg, neutral, pos]` probability vector. |
| `interpretation` | Classifier: `"Belongs to this Category"` / `"Does not Belong to this Category"`. Sentiment: human-readable valence label. |

**Examples**

```python
import sadbert

# Probe two categories with only the classifier head
result = sadbert.predict_individual_types(
    text=["honest", "lazy"],
    content_type=["Warmth", "Morality"],
    head_type="classifier",
)
print(result)

# Run both classifier and sentiment for one category
result = sadbert.predict_individual_types(
    text=["warm", "cold"],
    content_type="Warmth",
    head_type=["classifier", "sentiment"],
)
print(result)

# Run the sentiment head across all major categories
result = sadbert.predict_individual_types(
    text="aggressive",
    head_type="sentiment",   # content_type omitted → runs all applicable categories
)
print(result)

# Explicit pairs via manual_specification
result = sadbert.predict_individual_types(
    text=["honest", "deceptive"],
    manual_specification=[
        ("morality", "classifier"),
        ("morality", "sentiment"),
    ],
)
print(result)

# Unstacked — one DataFrame per input word
result = sadbert.predict_individual_types(
    text=["honest", "lazy"],
    content_type="Competence",
    head_type="classifier",
    stacked=False,
)
print(result["honest"])
```

Models are downloaded on first use and cached; only the heads you request are loaded into memory, making this significantly lighter than the full pipeline when you need targeted predictions.

---

### `sadbert.SADBERT(device=None, batch_size=32, load_models=True)`

Instantiate your own SADBERT object for full control.

```python
from sadbert import SADBERT

model = SADBERT(
    device="cuda",     # "cuda" | "mps" | "cpu" | None (auto-detect)
    batch_size=64,     # increase for faster throughput on GPU
    load_models=True,  # set False to defer model loading to first call
)

results = model.get_stereotype_content(["nurse", "engineer", "senator"])
```

---

## Model architecture

SADBERT uses a three-stage ensemble:

```
Input text
    │
    ▼
┌─────────────────────────────────────────────────────┐
│  Stage 1 · Master model (SADBERT_master_model)      │
│  Multi-label DistilBERT, 35 output classes          │
│  Softmax probabilities compared against per-class   │
│  Youden-J thresholds → candidate categories         │
└─────────────────────────────────────────────────────┘
    │  candidate categories
    ▼
┌─────────────────────────────────────────────────────┐
│  Stage 2 · Classifier heads (SADBERT_{cat}_classifier) │
│  One binary DistilBERT per category                 │
│  Veto gate — keeps only categories confirmed by     │
│  both master model and dedicated head               │
└─────────────────────────────────────────────────────┘
    │  confirmed categories + probabilities
    ▼
┌─────────────────────────────────────────────────────┐
│  Stage 3 · Sentiment models (SADBERT_{cat}_sentiment)  │
│  One 3-class DistilBERT per major category          │
│  Predicts negative / neutral / positive valence     │
└─────────────────────────────────────────────────────┘
    │
    ▼
  Results DataFrame
```

All models are hosted on HuggingFace at [huggingface.co/XanderD24](https://huggingface.co/XanderD24).

---





---

## Citation

If you use SADBERT in research, please cite the underlying SADCAT framework:

```
@misc{sadbert2025,
  author = {Deanhardt, Xander},
  title  = {{SADBERT}: Stereotype-content Analysis with {DistilBERT}},
  year   = {2025},
  url    = {https://github.com/XanderD24/sadbert},
}
```

---
## Citation of Original Work
All data used to fine-tune these models was taken from the SADCAT dictionary, published by Gandalf Nicolas, Xuecunzi Bai, and Susan T Fiske, and hosted on Github by Gandalf Nicolas. It was first published in this journal:

Nicolas, Gandalf, et al. “Comprehensive Stereotype Content Dictionaries Using a Semi‐Automated Method.” European Journal of Social Psychology, vol. 51, no. 1, Feb. 2021, pp. 178–196, https://doi.org/10.1002/ejsp.2724.

Github Link = {https://github.com/gandalfnicolas/SADCAT/tree/master}

OSF Repository = {https://osf.io/yx45f/}
## License

MIT License — see [LICENSE](LICENSE) for details.
