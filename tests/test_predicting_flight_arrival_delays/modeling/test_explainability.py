"""Tests for predicting_flight_arrival_delays.modeling.explainability.
"""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.frozen import FrozenEstimator
from sklearn.linear_model import LogisticRegression

pytest.importorskip("shap", reason="shap is not installed")

from predicting_flight_arrival_delays.modeling import (  # noqa: E402
    explainability as explainability_module,
)
from predicting_flight_arrival_delays.modeling.explainability import (  # noqa: E402
    _unwrap_calibration,
    explain_prediction,
    save_shap_waterfall_plot,
)


@pytest.fixture
def xy():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(
        {
            "signal": np.concatenate([rng.normal(-1, 0.5, 40), rng.normal(1, 0.5, 40)]),
            "noise": rng.normal(0, 1, 80),
        }
    )
    return X, pd.Series([0] * 40 + [1] * 40)


class TestUnwrapCalibration:
    def test_a_bare_estimator_is_returned_as_is(self, xy):
        X, y = xy
        model = LogisticRegression(max_iter=200).fit(X, y)

        assert _unwrap_calibration(model) is model

    def test_calibration_and_freezing_are_peeled_off(self, xy):
        """Explanations must reflect the base model, not the calibration wrapper."""
        X, y = xy
        base = LogisticRegression(max_iter=200).fit(X, y)
        calibrated = CalibratedClassifierCV(FrozenEstimator(base), method="isotonic").fit(X, y)

        assert _unwrap_calibration(calibrated) is base


class TestExplainPrediction:
    def test_empty_input_yields_no_explanation(self, xy):
        X, y = xy
        model = LogisticRegression(max_iter=200).fit(X, y)

        assert explain_prediction(model, X.head(0), "logistic_regression") == []

    def test_logistic_regression_uses_its_coefficients(self, xy):
        X, y = xy
        model = LogisticRegression(max_iter=200).fit(X, y)

        explanation = explain_prediction(model, X, "logistic_regression", top_k=2)

        assert [e["feature"] for e in explanation] == ["signal", "noise"]
        assert explanation[0]["value"] == pytest.approx(model.coef_[0][0])

    def test_top_k_limits_the_result(self, xy):
        X, y = xy
        model = LogisticRegression(max_iter=200).fit(X, y)

        assert len(explain_prediction(model, X, "logreg", top_k=1)) == 1

    def test_results_are_sorted_by_absolute_contribution(self, xy):
        X, y = xy
        model = RandomForestClassifier(n_estimators=10, random_state=0).fit(X, y)

        explanation = explain_prediction(model, X, "random_forest")
        values = [e["abs_value"] for e in explanation]

        assert values == sorted(values, reverse=True)

    def test_tree_models_go_through_shap(self, xy):
        X, y = xy
        model = RandomForestClassifier(n_estimators=10, random_state=0).fit(X, y)

        explanation = explain_prediction(model, X, "random_forest", top_k=2)

        assert {e["feature"] for e in explanation} == {"signal", "noise"}

    def test_an_unsupported_model_type_returns_nothing(self, xy):
        X, y = xy
        model = LogisticRegression(max_iter=200).fit(X, y)

        assert explain_prediction(model, X, "naive_bayes") == []


class TestDegradedInputs:
    """None of these may raise: an explanation is a nice-to-have, never a blocker."""

    def test_cross_validated_calibration_unwraps_to_the_base_estimator(self, xy):
        """Without an explicit FrozenEstimator there is still a wrapper to peel."""
        X, y = xy
        calibrated = CalibratedClassifierCV(
            LogisticRegression(max_iter=200), method="isotonic", cv=3
        ).fit(X, y)

        unwrapped = _unwrap_calibration(calibrated)

        assert isinstance(unwrapped, LogisticRegression)
        assert not isinstance(unwrapped, CalibratedClassifierCV)

    def test_a_model_without_coefficients_explains_nothing(self, xy):
        X, _ = xy

        class NoCoefficients:
            pass

        assert explain_prediction(NoCoefficients(), X, "logistic_regression") == []

    def test_a_coefficient_vector_shorter_than_the_features_is_truncated(self, xy):
        X, y = xy
        model = LogisticRegression(max_iter=200).fit(X, y)
        model.coef_ = np.array([[0.5]])

        explanation = explain_prediction(model, X, "logreg")

        assert len(explanation) == 1
        assert explanation[0]["feature"] == "signal"

    def test_a_non_tree_model_sent_down_the_shap_path_explains_nothing(self, xy):
        """TreeExplainer refuses it; the failure is logged, not raised."""
        X, y = xy
        model = LogisticRegression(max_iter=200).fit(X, y)

        assert explain_prediction(model, X, "random_forest") == []

    def test_an_unfitted_tree_model_explains_nothing(self, xy):
        X, _ = xy

        assert explain_prediction(RandomForestClassifier(), X, "lightgbm") == []

    def test_model_type_matching_ignores_case(self, xy):
        X, y = xy
        model = LogisticRegression(max_iter=200).fit(X, y)

        assert explain_prediction(model, X, "Logistic_Regression", top_k=1)


class TestShapValueShapes:
    """SHAP returns a different array layout per model family, so the dispatch
    on ndim and axis order is what decides whether the right class is read."""

    @pytest.fixture
    def fake_explainer(self, monkeypatch):
        def install(values):
            class FakeExplainer:
                def __init__(self, model):
                    pass

                def __call__(self, x):
                    return SimpleNamespace(values=np.asarray(values))

            monkeypatch.setattr(
                explainability_module.shap, "TreeExplainer", FakeExplainer
            )

        return install

    @pytest.fixture
    def tree_model(self, xy):
        X, y = xy
        return RandomForestClassifier(n_estimators=5, random_state=0).fit(X, y)

    def test_two_dimensional_values_are_read_directly(self, xy, tree_model, fake_explainer):
        """LightGBM's binary explainer returns (samples, features)."""
        X, _ = xy
        fake_explainer([[0.4, -0.1]])

        explanation = explain_prediction(tree_model, X, "lightgbm")

        assert [e["value"] for e in explanation] == [0.4, -0.1]

    def test_features_on_the_middle_axis(self, xy, tree_model, fake_explainer):
        """(samples, features, classes): the positive class is the last axis."""
        X, _ = xy
        fake_explainer([[[0.1, 0.9], [0.2, -0.8]]])

        explanation = explain_prediction(tree_model, X, "random_forest")

        assert [e["value"] for e in explanation] == [0.9, -0.8]

    def test_features_on_the_last_axis(self, xy, tree_model, fake_explainer):
        """(samples, classes, features): the positive class is the middle axis."""
        X, _ = xy
        fake_explainer([[[0.1, 0.2], [0.9, -0.8], [0.0, 0.0]]])

        explanation = explain_prediction(tree_model, X, "random_forest")

        assert [e["value"] for e in explanation] == [0.9, -0.8]

    def test_a_single_output_model_reads_the_only_class(self, xy, tree_model, fake_explainer):
        X, _ = xy
        fake_explainer([[[0.5], [-0.3]]])

        explanation = explain_prediction(tree_model, X, "random_forest")

        assert [e["value"] for e in explanation] == [0.5, -0.3]

    def test_fewer_values_than_features_are_truncated(self, xy, tree_model, fake_explainer):
        """A mismatch is reported and trimmed rather than raising an IndexError."""
        X, _ = xy
        fake_explainer([[0.4]])

        explanation = explain_prediction(tree_model, X, "lightgbm")

        assert [e["feature"] for e in explanation] == ["signal"]

    def test_a_layout_matching_no_axis_explains_nothing(self, xy, tree_model, fake_explainer):
        X, _ = xy
        fake_explainer(np.zeros((1, 5, 7)))

        assert explain_prediction(tree_model, X, "random_forest") == []

    def test_an_unexpected_dimensionality_explains_nothing(self, xy, tree_model, fake_explainer):
        X, _ = xy
        fake_explainer(np.zeros((1, 2, 3, 4)))

        assert explain_prediction(tree_model, X, "random_forest") == []


class TestSaveShapWaterfallPlot:
    def test_a_plot_is_written_for_tree_models(self, tmp_path, xy):
        X, y = xy
        model = RandomForestClassifier(n_estimators=10, random_state=0).fit(X, y)

        out = save_shap_waterfall_plot(model, X, "random_forest", tmp_path / "p" / "shap.png")

        assert out is not None
        assert out.exists()

    def test_non_tree_models_are_skipped_not_failed(self, tmp_path, xy):
        X, y = xy
        model = LogisticRegression(max_iter=200).fit(X, y)

        assert save_shap_waterfall_plot(model, X, "logreg", tmp_path / "shap.png") is None

    def test_empty_input_is_skipped(self, tmp_path, xy):
        X, y = xy
        model = RandomForestClassifier(n_estimators=10, random_state=0).fit(X, y)

        assert (
            save_shap_waterfall_plot(model, X.head(0), "random_forest", tmp_path / "s.png")
            is None
        )

    def test_an_explainer_that_cannot_be_built_is_skipped(self, tmp_path, xy):
        """Declared tree-based but not actually a tree: logged, not raised."""
        X, y = xy
        model = LogisticRegression(max_iter=200).fit(X, y)

        assert save_shap_waterfall_plot(model, X, "random_forest", tmp_path / "s.png") is None

    def test_lightgbm_plots_are_supported_too(self, tmp_path, xy):
        from lightgbm import LGBMClassifier

        X, y = xy
        model = LGBMClassifier(n_estimators=10, verbose=-1, random_state=0).fit(X, y)

        out = save_shap_waterfall_plot(model, X, "lightgbm", tmp_path / "lgbm.png")

        assert out is not None and out.exists()

    def test_a_failure_while_writing_the_figure_is_swallowed(self, tmp_path, xy, monkeypatch):
        """A plot that cannot be saved must not take the caller down with it."""
        X, y = xy
        model = RandomForestClassifier(n_estimators=5, random_state=0).fit(X, y)

        def boom(*args, **kwargs):
            raise OSError("disk full")

        monkeypatch.setattr(explainability_module.plt, "savefig", boom)

        assert save_shap_waterfall_plot(model, X, "random_forest", tmp_path / "s.png") is None
