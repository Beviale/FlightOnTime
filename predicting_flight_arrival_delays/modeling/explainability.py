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

# Tree-based models explained via shap.TreeExplainer. "lightgbm" added here: the
# original only listed random_forest/decision_tree, which silently fell through
# to no explanation (implicit None return) for this project's main algorithm.
TREE_MODEL_TYPES = ("random_forest", "decision_tree", "lightgbm")


def _unwrap_calibration(model: Any) -> Any:
    """Return the fitted base estimator, unwrapping calibration/freezing if present.

    modeling/train.py wraps the fitted base estimator in
    CalibratedClassifierCV(FrozenEstimator(base)) when calibrate=True. Neither
    TreeExplainer nor raw model.coef_ access work on that wrapper directly.
    Calibration is a monotonic rescaling of the output probability, so it does not
    change which features drive a prediction or their relative ranking, only the
    absolute probability scale -- explaining the pre-calibration model is
    equivalent for this purpose.

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
    """
    Build a explanation for a single sample.
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

        if X.empty:
            logger.warning("Received empty DataFrame for SHAP explanation; returning empty list.")
            return []

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
    """
    Save a SHAP waterfall plot for a single sample to the given output path.
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
            # Same dynamic shape check as explain_prediction, kept consistent here
            # (the original picked class 1 unconditionally, which breaks for
            # single-output shapes where there is no class axis to index).
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