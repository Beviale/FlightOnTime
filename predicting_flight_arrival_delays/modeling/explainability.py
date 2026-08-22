"""Per-prediction explanations for the flight-delay classifiers."""
from __future__ import annotations
from pathlib import Path
from typing import Any
import matplotlib
matplotlib.use("Agg")
from loguru import logger
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator

# Tree-based models explained via shap.TreeExplainer.
TREE_MODEL_TYPES = ("random_forest", "decision_tree", "lightgbm")


def _unwrap_calibration(model: Any) -> Any:
    """Return the fitted base estimator, unwrapping calibration/freezing if present.

    Args:
        model: A fitted estimator, possibly CalibratedClassifierCV(FrozenEstimator(base)).

    Returns:
        The innermost fitted base estimator (e.g. the LGBMClassifier itself).
    """
    if isinstance(model, CalibratedClassifierCV):
        inner = model.calibrated_classifiers_[0].estimator
        if isinstance(inner, FrozenEstimator):
            return inner.estimator
        return inner
    return model


def explain_prediction(
    model: Any,
    X: pd.DataFrame,
    model_type: str,
    top_k: int = 5,
):
    """Build a feature-level explanation for a single sample's prediction.

    Uses coefficients for logistic regression, or SHAP TreeExplainer for
    tree-based models (random forest, decision tree, lightgbm). Calibration
    wrappers (CalibratedClassifierCV/FrozenEstimator) are unwrapped first, so
    the explanation reflects the underlying base estimator. Only the first
    row of X is explained.

    Args:
        model: A fitted estimator, possibly wrapped in
            CalibratedClassifierCV(FrozenEstimator(base_estimator)).
        X: Features to explain; only the first row (X.iloc[[0]]) is used.
        model_type: Which explanation strategy to use. "logreg"/
            "logistic_regression" for coefficient-based explanations; one of
            TREE_MODEL_TYPES ("random_forest", "decision_tree", "lightgbm")
            for SHAP-based explanations. Case-insensitive.
        top_k: How many top features (by absolute contribution) to return.

    Returns:
        Up to top_k dicts, sorted by absolute contribution descending, each
        with:
            - "feature": Column name.
            - "value": Signed contribution (coefficient or SHAP value).
            - "abs_value": Its absolute value, used for ranking.
        An empty list if X is empty, model_type is unsupported, the model
        lacks coef_ (logistic regression path), or SHAP explanation fails -
        never raises for these cases, only logs a warning/error.
    """

    if X.empty:
        logger.warning("Received empty DataFrame for explanation; returning empty list.")
        return []

    model_type = model_type.lower()
    model = _unwrap_calibration(model)
    x = X.iloc[[0]]
    feature_names = x.columns.tolist()

    # ---------------------------------------------------------------------
    # 1) Logistic Regression → use coefficients
    # ---------------------------------------------------------------------
    if model_type in ("logreg", "logistic_regression"):
        logger.info("Using coefficient-based explanation for Logistic Regression.")

        if not hasattr(model, "coef_"):
            logger.error(
                "Model has no coef_ attribute;cannot build coefficient-based explanation."
            )
            return []

        coef = np.asarray(model.coef_[0]).reshape(-1)
        if coef.shape[0] != len(feature_names):
            logger.warning(
                f"Coefficient vector length ({coef.shape[0]}) does not match "
                f"number of features ({len(feature_names)}). "
                "Truncating to minimum length."
            )

        n = min(len(feature_names), coef.shape[0])
        explanations = [
            {
                "feature": feature_names[i],
                "value": float(coef[i]),
                "abs_value": float(abs(coef[i])),
            }
            for i in range(n)
        ]

        explanations = sorted(explanations, key=lambda d: d["abs_value"], reverse=True)[:top_k]
        logger.info(
            f"Built coefficient-based explanation. Returning top {len(explanations)} features."
        )
        return explanations

    # ---------------------------------------------------------------------
    # 2) Tree-based models → SHAP TreeExplainer
    # ---------------------------------------------------------------------
    if model_type in TREE_MODEL_TYPES:
        logger.info(f"Using SHAP TreeExplainer for tree-based model ('{model_type}').")

        x = X.iloc[[0]]
        feature_names = x.columns.tolist()

        try:
            explainer = shap.TreeExplainer(model)
            shap_exp = explainer(x)
            values = np.asarray(shap_exp.values)
            logger.debug(f"Raw SHAP values shape: {values.shape!r}")
        except Exception as e:
            logger.error(f"SHAP TreeExplainer failed: {e}")
            logger.warning("SHAP explanation not available for this model.")
            return []

        if values.ndim == 2:
            shap_vec = values[0]

        elif values.ndim == 3:
            n_samples, dim2, dim3 = values.shape

            if dim2 == x.shape[1]:
                n_outputs = dim3
                class_index = 1 if n_outputs > 1 else 0
                shap_vec = values[0, :, class_index]

            elif dim3 == x.shape[1]:
                n_outputs = dim2
                class_index = 1 if n_outputs > 1 else 0
                shap_vec = values[0, class_index, :]

            else:
                logger.error(f"Unexpected SHAP shape {values.shape} for {x.shape[1]} features.")
                return []

        else:
            logger.error(f"Unexpected SHAP values dimension: {values.ndim}")
            return []

        shap_vec = np.asarray(shap_vec).reshape(-1)

        if shap_vec.shape[0] != len(feature_names):
            logger.warning(
                f"SHAP vector length ({shap_vec.shape[0]}) "
                f"!= number of features ({len(feature_names)}). "
                "Truncating to minimum length."
            )

        n = min(len(feature_names), shap_vec.shape[0])
        explanations = [
            {
                "feature": feature_names[i],
                "value": float(shap_vec[i]),
                "abs_value": float(abs(shap_vec[i])),
            }
            for i in range(n)
        ]

        explanations = sorted(explanations, key=lambda d: d["abs_value"], reverse=True)[:top_k]

        logger.info(f"Built SHAP-based explanation. Returning top {len(explanations)} features.")
        return explanations

    # ---------------------------------------------------------------------
    # 3) Unknown model_type → explicit empty result instead of an implicit None
    # ---------------------------------------------------------------------
    logger.warning(f"Unsupported model_type '{model_type}'; no explanation available.")
    return []


def save_shap_waterfall_plot(
    model: Any,
    X: pd.DataFrame,
    model_type: str,
    output_path: Path,
) -> Path | None:
    """Save a SHAP waterfall plot for a single sample's prediction to disk.

    Only supported for tree-based models (TREE_MODEL_TYPES); calibration
    wrappers are unwrapped first, same as explain_prediction. Only the first
    row of X is plotted. Parent directories of output_path are created if
    missing.

    Args:
        model: A fitted estimator, possibly wrapped in
            CalibratedClassifierCV(FrozenEstimator(base_estimator)).
        X: Features to plot; only the first row (X.iloc[[0]]) is used.
        model_type: The model's type, case-insensitive. Must be one of
            TREE_MODEL_TYPES for a plot to be generated; anything else is
            skipped (not an error).
        output_path: Where to save the plot (e.g. a .png path). Parent
            directories are created automatically.

    Returns:
        output_path on success. None if model_type is not tree-based, X is
        empty, the SHAP explainer could not be built, or saving the figure
        failed - never raises, only logs a warning/error in these cases.
    """
    model_type = model_type.lower()
    model = _unwrap_calibration(model)

    if model_type not in TREE_MODEL_TYPES:
        logger.warning(
            f"Waterfall plot is only supported for tree-based models. "
            f"Got model_type='{model_type}'. Skipping plot generation."
        )
        return None

    if X.empty:
        logger.warning("Received empty DataFrame for SHAP plot; skipping.")
        return None

    x = X.iloc[[0]]
    logger.info(f"Generating SHAP waterfall plot for model_type='{model_type}'.")

    try:
        explainer = shap.TreeExplainer(model)
        shap_exp = explainer(x)
    except Exception as e:
        logger.error(f"Failed to build SHAP explainer for plot: {e}")
        return None

    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)

        shap_to_plot = shap_exp
        vals = np.asarray(shap_exp.values)
        if vals.ndim == 3:
            if vals.shape[1] == x.shape[1]:
                class_index = 1 if vals.shape[2] > 1 else 0
                shap_to_plot = shap_exp[..., class_index]
            elif vals.shape[2] == x.shape[1]:
                class_index = 1 if vals.shape[1] > 1 else 0
                shap_to_plot = shap_exp[:, class_index, :]
            else:
                logger.warning(
                    f"Unexpected shape for SHAP values in plot: {vals.shape}. "
                    "Falling back to shap_exp[0]."
                )
                shap_to_plot = shap_exp

        plt.figure()
        shap.plots.waterfall(shap_to_plot[0], show=False)
        plt.tight_layout()
        plt.savefig(output_path, bbox_inches="tight")
        plt.close()

        logger.success(f"SHAP waterfall plot saved to {output_path}")
        return output_path
    except Exception as e:
        logger.error(f"Failed to save SHAP waterfall plot: {e}")
        return None