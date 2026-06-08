"""Signed mean SHAP effects for tree pipeline models (optional ``shap`` extra)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline


def shap_available() -> bool:
    try:
        import shap  # noqa: F401

        return True
    except ImportError:
        return False


def _is_tree_pipeline(pipeline: Pipeline) -> bool:
    model = pipeline.named_steps.get("model")
    if model is None:
        return False
    return hasattr(model, "estimators_") or hasattr(model, "get_booster")


def _clean_feature_name(name: str) -> str:
    return name.split("__", 1)[-1] if "__" in name else name


def compute_mean_shap_effects(
    pipeline: Any,
    df: pd.DataFrame,
    target: str,
    feature_cols: list[str],
    *,
    max_samples: int = 512,
    random_state: int = 42,
) -> dict[str, float] | None:
    """
    Mean signed SHAP value per preprocessed feature (tree models only).

    Positive values push average predictions up; negative values push them down.
    Returns ``None`` if ``shap`` is not installed or the pipeline is not a tree model.
    """
    if not isinstance(pipeline, Pipeline) or not _is_tree_pipeline(pipeline):
        return None
    if not shap_available():
        return None

    import shap

    from utils.training_matrix import prep_xy

    prep = pipeline.named_steps["prep"]
    model = pipeline.named_steps["model"]
    X, _ = prep_xy(df, target, feature_cols)
    if X.empty:
        return None
    if len(X) > max_samples:
        X = X.sample(n=max_samples, random_state=random_state)
    X_proc = prep.transform(X)

    explainer = shap.TreeExplainer(model)
    values = explainer.shap_values(X_proc, check_additivity=False)
    if isinstance(values, list):
        values = values[0]
    means = np.asarray(values, dtype=float).mean(axis=0)
    names = [_clean_feature_name(str(n)) for n in prep.get_feature_names_out()]
    return {name: float(v) for name, v in zip(names, means)}


def format_top_shap_effects(effects: dict[str, float], *, top_n: int = 6) -> str:
    pairs = sorted(effects.items(), key=lambda x: abs(x[1]), reverse=True)[:top_n]
    short = lambda n: n if len(n) <= 36 else n[:35] + "…"
    return ", ".join(f"{short(k)}={v:+.3g}" for k, v in pairs)
