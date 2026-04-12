"""
Unit tests for the SADBERT package.

These tests verify the package's structure, data loading, and output
contract without requiring the HuggingFace models to be downloaded
(model-loading tests are marked with pytest.mark.slow and skipped by
default in CI to avoid network calls).

Run all tests:
    pytest

Run only fast (offline) tests:
    pytest -m "not slow"

Run including model-download tests:
    pytest -m slow
"""

import math
import pickle
from pathlib import Path

import pandas as pd
import pytest

# ── Path to bundled data ──────────────────────────────────────────────────────
_DATA_DIR = Path(__file__).parent.parent / "sadbert" / "data"


# ─────────────────────────────────────────────────────────────────────────────
# Data file tests (always run — no network required)
# ─────────────────────────────────────────────────────────────────────────────
class TestBundledData:
    def test_label_mappings_exists(self):
        assert (_DATA_DIR / "label_mappings.pkl").exists(), \
            "label_mappings.pkl missing from sadbert/data/"

    def test_interpretation_dict_exists(self):
        assert (_DATA_DIR / "interpretation_dict.pkl").exists(), \
            "interpretation_dict.pkl missing from sadbert/data/"

    def test_roc_dict_exists(self):
        assert (_DATA_DIR / "ROC_dict.pkl").exists(), (
            "ROC_dict.pkl missing from sadbert/data/.\n"
            "Copy it there before running tests: cp /path/to/ROC_dict.pkl sadbert/data/"
        )

    def test_label_mappings_structure(self):
        with open(_DATA_DIR / "label_mappings.pkl", "rb") as f:
            lm = pickle.load(f)
        assert "labeltoid" in lm
        assert "idtolabel" in lm
        # At least 35 valid (non-nan) entries
        valid = {k: v for k, v in lm["idtolabel"].items()
                 if v is not None
                 and not (isinstance(v, float) and math.isnan(v))
                 and str(v) != "nan"}
        assert len(valid) >= 35

    def test_interpretation_dict_structure(self):
        with open(_DATA_DIR / "interpretation_dict.pkl", "rb") as f:
            interp = pickle.load(f)
        # Must have all 13 major categories
        from sadbert import MAJOR_CATS
        for cat in MAJOR_CATS:
            assert cat in interp, f"'{cat}' missing from interpretation_dict"
            # Each entry maps {0, 1, 2} → str
            for label in (0, 1, 2):
                assert label in interp[cat]
                assert isinstance(interp[cat][label], str)


# ─────────────────────────────────────────────────────────────────────────────
# Package structure tests
# ─────────────────────────────────────────────────────────────────────────────

class TestPackageStructure:
    def test_imports(self):
        import sadbert
        assert hasattr(sadbert, "SADBERT")
        assert hasattr(sadbert, "get_stereotype_content")
        assert hasattr(sadbert, "ALL_CATS")
        assert hasattr(sadbert, "MAJOR_CATS")
        assert hasattr(sadbert, "MINOR_CATS")

    def test_category_lists(self):
        from sadbert import ALL_CATS, MAJOR_CATS, MINOR_CATS
        assert len(MAJOR_CATS) == 13
        assert len(MINOR_CATS) == 22
        assert len(ALL_CATS)   == 35
        assert set(ALL_CATS) == set(MAJOR_CATS) | set(MINOR_CATS)

    def test_version(self):
        import sadbert
        assert hasattr(sadbert, "__version__")
        assert isinstance(sadbert.__version__, str)


# ─────────────────────────────────────────────────────────────────────────────
# Model tests — require network / large download; skipped in fast mode
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.slow
class TestSADBERTModel:
    @pytest.fixture(scope="class")
    def model(self):
        from sadbert import SADBERT
        return SADBERT(device="cpu", batch_size=8)

    def test_single_string_returns_dataframe(self, model):
        result = model.get_stereotype_content("honest")
        assert isinstance(result, pd.DataFrame)

    def test_dataframe_columns_stacked(self, model):
        result = model.get_stereotype_content("honest", stacked=True)
        expected = {"text", "category", "probability",
                    "valence", "valence probability", "interpretation"}
        assert expected.issubset(set(result.columns))

    def test_dataframe_columns_unstacked_single(self, model):
        result = model.get_stereotype_content("honest", stacked=False)
        assert isinstance(result, pd.DataFrame)
        expected = {"category", "probability",
                    "valence", "valence probability", "interpretation"}
        assert expected.issubset(set(result.columns))

    def test_list_unstacked_returns_dict(self, model):
        result = model.get_stereotype_content(["honest", "lazy"], stacked=False)
        assert isinstance(result, dict)
        assert "honest" in result
        assert "lazy" in result
        for df in result.values():
            assert isinstance(df, pd.DataFrame)

    def test_list_stacked_returns_dataframe(self, model):
        result = model.get_stereotype_content(["honest", "lazy"], stacked=True)
        assert isinstance(result, pd.DataFrame)
        assert "text" in result.columns
        assert set(result["text"].unique()).issubset({"honest", "lazy"})

    def test_none_category_for_unknown_word(self, model):
        # 'zymurgy' is an unlikely stereotype word — may return "None"
        result = model.get_stereotype_content("zymurgy", stacked=False)
        if result["category"].iloc[0] == "None":
            assert result["valence"].iloc[0] == "None"
            assert result["interpretation"].iloc[0] == "None"

    def test_minor_categories_have_no_valence(self, model):
        from sadbert import MINOR_CATS
        result = model.get_stereotype_content(["dress", "village"], stacked=True)
        minor_rows = result[result["category"].isin(MINOR_CATS)]
        if not minor_rows.empty:
            assert (minor_rows["valence"] == "None").all()
            assert (minor_rows["valence probability"] == "None").all()
            assert (minor_rows["interpretation"] == "None").all()

    def test_module_level_function(self):
        import sadbert
        result = sadbert.get_stereotype_content("honest")
        assert isinstance(result, pd.DataFrame)
