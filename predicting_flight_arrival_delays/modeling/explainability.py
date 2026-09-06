"""Per-prediction explanations for the flight-delay classifiers.

Every explanation here is an additive decomposition: a base value plus one
contribution per column, summing back to what the model said. That identity is what
lets the serving path draw a waterfall that closes on the probability it reported,
and it holds for both families - SHAP values for the trees, the closed form for the
linear models, which are already a sum of contributions.

The columns those contributions arrive on are the model's, though, where an airport
is three hundred one-hot columns. Reading them out as they come would name none of
what a caller recognises, so they are folded back onto the request columns before
anyone sees them.
"""

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
    feature_means: dict[str, float] | None = None,
):
    """Build a feature-level explanation for a single sample's prediction.

    Two paths, both additive. SHAP TreeExplainer for the tree models (random
    forest, decision tree, lightgbm), and the closed form for logistic regression,
    where the model is already a sum of contributions and the Shapley values can be
    read straight off it. Calibration wrappers (CalibratedClassifierCV/
    FrozenEstimator) are unwrapped first, so the explanation reflects the underlying
    base estimator. Only the first row of X is explained.

    Args:
        model: A fitted estimator, possibly wrapped in
            CalibratedClassifierCV(FrozenEstimator(base_estimator)).
        X: Features to explain; only the first row (X.iloc[[0]]) is used.
        model_type: Which explanation strategy to use. "logreg"/
            "logistic_regression" for the linear path; one of TREE_MODEL_TYPES
            ("random_forest", "decision_tree", "lightgbm") for the SHAP path.
            Case-insensitive.
        top_k: How many top features (by absolute contribution) to return.
        feature_means: The average training flight on these columns.

    Returns:
        Up to top_k dicts, sorted by absolute contribution descending, each
        with:
            - "feature": Column name.
            - "value": Signed contribution.
            - "abs_value": Its absolute value, used for ranking.
        An empty list if X is empty, model_type is unsupported, the model
        lacks coef_ (linear path), or the SHAP explanation fails -
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
    # 1) Logistic Regression
    # ---------------------------------------------------------------------
    if model_type in ("logreg", "logistic_regression"):
        logger.info("Explaining a linear model in closed form.")

        if not hasattr(model, "coef_"):
            logger.error("Model has no coef_ attribute; cannot explain a linear model.")
            return []

        coef = np.asarray(model.coef_[0]).reshape(-1)
        if coef.shape[0] != len(feature_names):
            logger.warning(
                f"Coefficient vector length ({coef.shape[0]}) does not match "
                f"number of features ({len(feature_names)}). "
                "Truncating to minimum length."
            )

        n = min(len(feature_names), coef.shape[0])

        values = np.asarray(x.to_numpy(), dtype=float).reshape(-1)
        if feature_means:
            centre = np.array(
                [float(feature_means.get(name, 0.0)) for name in feature_names],
                dtype=float,
            )
            values = values - centre
        contributions = coef[:n] * values[:n]
        explanations = [
            {
                "feature": feature_names[i],
                "value": float(contributions[i]),
                "abs_value": float(abs(contributions[i])),
            }
            for i in range(n)
        ]

        explanations = sorted(explanations, key=lambda d: d["abs_value"], reverse=True)[:top_k]
        logger.info(f"Explained a linear model. Returning top {len(explanations)} features.")
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
            _, dim2, dim3 = values.shape

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


def _encoded_importance(model: Any, columns: list[str]) -> dict[str, float]:
    """How much the estimator leans on each of its own input columns.

    Args:
        model: A fitted estimator, possibly wrapped in calibration.
        columns: The column names the model was fitted on, in order.

    Returns:
        One weight per column, or an empty dict if the estimator exposes neither
        importances nor coefficients, or reports a different number of them.
    """
    base = _unwrap_calibration(model)

    if hasattr(base, "feature_importances_"):
        weights = np.asarray(base.feature_importances_, dtype=float)
    elif hasattr(base, "coef_"):
        weights = np.abs(np.asarray(base.coef_, dtype=float)).ravel()
    else:
        logger.warning(f"{type(base).__name__} reports no importances - skipping.")
        return {}

    if len(weights) != len(columns):
        logger.warning(
            f"{type(base).__name__} reports {len(weights)} importances for "
            f"{len(columns)} columns - skipping."
        )
        return {}

    return dict(zip(columns, weights, strict=True))


def _ancestor(column: str, raw_columns: list[str], transformer: Any) -> str | None:
    """Trace one of the model's columns back to the request column it came from.

    The inverse of what encoding did, and it has to agree with app.inputs.contributes:
    a column is either carried through as itself, one-hot expanded, or turned into a
    delay rate.

    Args:
        column: A column the model was fitted on.
        raw_columns: The request columns it could descend from, longest first.

    Returns:
        The request column it came from, or None if it came from something the
        caller never sends - the weather, for instance.
    """
    if column in raw_columns:
        return column

    for raw in raw_columns:
        if column == f"{raw}DelayRate" and raw in transformer.delay_rate_columns:
            return raw
        if column.startswith(f"{raw}_") and raw in transformer.category_keep:
            return raw

    return None


def request_column_importance(
    model: Any, columns: list[str], transformer: Any, raw_columns: list[str]
) -> dict[str, float]:
    """Rank the columns a caller sends by how much the model leans on them.

    A request column rarely reaches the model as itself: it arrives as a block of
    one-hot columns, or as a delay rate, or both. Its weight here is the sum of what
    its descendants carry.

    Args:
        model: The fitted estimator.
        columns: The column names it was fitted on, in order.
        transformer: The fitted transformer that produced them.
        raw_columns: The columns a caller can send, from app.inputs.

    Returns:
        Request column to its share of the total weight, largest first. Shares sum
        to at most 1: what the service supplies by itself - the weather - is left
        out, since a caller cannot be warned about a column they do not send.
        Empty if the estimator reports no importances.
    """
    encoded = _encoded_importance(model, columns)
    if not encoded:
        return {}

    by_length = sorted(raw_columns, key=len, reverse=True)
    total = float(sum(encoded.values())) or 1.0

    folded: dict[str, float] = {}
    for column, weight in encoded.items():
        raw = _ancestor(column, by_length, transformer)
        if raw is not None:
            folded[raw] = folded.get(raw, 0.0) + float(weight) / total

    return dict(sorted(folded.items(), key=lambda item: item[1], reverse=True))


def request_column_contributions(
    model: Any,
    X: pd.DataFrame,
    transformer: Any,
    model_type: str,
    top_k: int = 5,
    feature_means: dict[str, float] | None = None,
) -> list[dict[str, Any]]:
    """Explain one prediction in terms of the columns a person recognises.

    Args:
        model: The fitted estimator that produced the prediction.
        X: The prepared matrix; only its first row is explained.
        transformer: The fitted transformer that built the columns.
        model_type: "logistic_regression" or one of the tree types.
        top_k: How many columns to report.
        feature_means: The average training flight, passed through so a linear
            model measures each column against it.

    Returns:
        Up to top_k dicts, largest absolute effect first, each with "column" and
        "contribution" - positive towards a delay, negative towards an on-time
        arrival. Empty if the explanation could not be built.
    """
    detailed = explain_prediction(
        model, X, model_type, top_k=len(X.columns), feature_means=feature_means
    )
    if not detailed:
        return []

    sources = sorted(
        set(transformer.category_keep) | set(transformer.delay_rate_columns),
        key=len,
        reverse=True,
    )

    folded: dict[str, float] = {}
    for item in detailed:
        column = _ancestor(item["feature"], sources, transformer) or item["feature"]
        folded[column] = folded.get(column, 0.0) + float(item["value"])

    ranked = sorted(folded.items(), key=lambda item: abs(item[1]), reverse=True)
    return [
        {"column": column, "contribution": contribution} for column, contribution in ranked[:top_k]
    ]


def _logit(p: float) -> float:
    """The log-odds of a probability, kept away from the infinities at the ends."""
    p = float(np.clip(p, 1e-6, 1 - 1e-6))
    return float(np.log(p / (1 - p)))


def _base_value(
    model: Any, model_type: str, feature_means: dict[str, float] | None = None
) -> float | None:
    """What the model answers before it knows anything about this flight.

    The point a waterfall starts from. Without it the steps have no origin, and a
    chart could only show them floating around zero.

    Args:
        model: A fitted estimator, possibly wrapped in calibration.
        model_type: "logistic_regression" or one of the tree types.
        feature_means: The average training flight.

    Returns:
        The base value in the same units as the contributions, or None if it cannot
        be read.
    """
    base = _unwrap_calibration(model)
    model_type = model_type.lower()

    if model_type in ("logreg", "logistic_regression"):
        intercept = getattr(base, "intercept_", None)
        if intercept is None:
            return None
        start = float(np.ravel(intercept)[0])
        if feature_means:
            coef = np.ravel(base.coef_)
            centre = np.array(
                [
                    float(feature_means.get(c, 0.0))
                    for c in getattr(base, "feature_names_in_", list(feature_means))
                ],
                dtype=float,
            )
            start += float(np.dot(coef[: len(centre)], centre[: len(coef)]))
        return start

    if model_type in TREE_MODEL_TYPES:
        try:
            expected = shap.TreeExplainer(base).expected_value
        except Exception as e:
            logger.error(f"Could not read the SHAP base value: {e}")
            return None
        expected = np.ravel(expected)
        return float(expected[1] if expected.size > 1 else expected[0])

    return None


def waterfall_terms(
    model: Any,
    X: pd.DataFrame,
    transformer: Any,
    model_type: str,
    served_probability: float,
    top_k: int = 5,
    feature_means: dict[str, float] | None = None,
) -> dict[str, Any] | None:
    """Everything a waterfall needs to close on the probability that was served.

    Args:
        model: The fitted estimator that produced the prediction.
        X: The prepared matrix; only its first row is explained.
        transformer: The fitted transformer that built the columns.
        model_type: "logistic_regression" or one of the tree types.
        served_probability: The probability the caller was given, calibration
            included.
        top_k: How many columns to report on their own.

    Returns:
        The base value, the leading contributions, the summed rest and the
        calibration step - or None if no explanation could be built.
    """
    folded = request_column_contributions(
        model,
        X,
        transformer,
        model_type,
        top_k=len(X.columns),
        feature_means=feature_means,
    )
    base = _base_value(model, model_type, feature_means)
    if not folded or base is None:
        return None

    total = base + sum(item["contribution"] for item in folded)

    raw = float(_unwrap_calibration(model).predict_proba(X.iloc[[0]])[0, 1])
    in_log_odds = abs(total - _logit(raw)) <= abs(total - raw)
    link = _logit if in_log_odds else float

    leading = folded[:top_k]
    return {
        "base_value": base,
        "contributions": leading,
        "other_contribution": sum(item["contribution"] for item in folded[top_k:]),
        "calibration": link(served_probability) - total,
        "log_odds": in_log_odds,
    }
