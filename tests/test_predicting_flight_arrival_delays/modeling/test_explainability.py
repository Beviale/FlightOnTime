"""Tests for predicting_flight_arrival_delays.modeling.explainability."""

from types import SimpleNamespace
from typing import ClassVar

import numpy as np
import pandas as pd
import pytest
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.frozen import FrozenEstimator
from sklearn.linear_model import LogisticRegression

pytest.importorskip("shap", reason="shap is not installed")

from predicting_flight_arrival_delays.modeling import (
    explainability as explainability_module,
)
from predicting_flight_arrival_delays.modeling.explainability import (
    _unwrap_calibration,
    explain_prediction,
    request_column_contributions,
    request_column_importance,
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

        by_name = {e["feature"]: e["value"] for e in explanation}
        assert by_name["signal"] == pytest.approx(model.coef_[0][0] * X["signal"].iloc[0])

    def test_a_logistic_explanation_differs_between_rows(self, xy):
        X, y = xy
        model = LogisticRegression(max_iter=200).fit(X, y)

        first = explain_prediction(model, X.iloc[[0]], "logreg", top_k=2)
        second = explain_prediction(model, X.iloc[[1]], "logreg", top_k=2)

        assert [e["value"] for e in first] != [e["value"] for e in second]

    def test_a_column_the_flight_does_not_have_contributes_nothing(self):
        X = pd.DataFrame({"a_x": [1.0, 0.0, 1.0, 0.0], "a_y": [0.0, 1.0, 0.0, 1.0]})
        y = pd.Series([1, 0, 1, 0])
        model = LogisticRegression(max_iter=200).fit(X, y)

        explanation = explain_prediction(model, X.iloc[[0]], "logreg", top_k=2)

        assert (
            dict(zip([e["feature"] for e in explanation], [e["value"] for e in explanation]))[
                "a_y"
            ]
            == 0.0
        )

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

            monkeypatch.setattr(explainability_module.shap, "TreeExplainer", FakeExplainer)

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
            save_shap_waterfall_plot(model, X.head(0), "random_forest", tmp_path / "s.png") is None
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


class StubTransformer:
    def __init__(self, category_keep=None, delay_rate_columns=()):
        self.category_keep = category_keep or {}
        self.delay_rate_columns = list(delay_rate_columns)


class TestRequestColumnImportance:
    """What the model leans on, folded back onto the columns a caller sends."""

    RAW: ClassVar[list] = [
        "OriginAirportID",
        "OriginCarrier",
        "OriginCongestion",
        "Dest",
        "DestCarrier",
    ]

    def a_model(self, weights):
        return SimpleNamespace(feature_importances_=np.array(weights, dtype=float))

    def test_a_one_hot_block_counts_as_its_source_column(self):
        columns = ["OriginAirportID_10397", "OriginAirportID_11298", "OriginCongestion"]
        transformer = StubTransformer(category_keep={"OriginAirportID": {"10397", "11298"}})

        importance = request_column_importance(
            self.a_model([0.2, 0.2, 0.6]), columns, transformer, self.RAW
        )

        assert importance["OriginAirportID"] == pytest.approx(0.4)
        assert importance["OriginCongestion"] == pytest.approx(0.6)

    def test_a_delay_rate_counts_as_the_column_it_came_from(self):
        columns = ["OriginCarrierDelayRate", "OriginCongestion"]
        transformer = StubTransformer(delay_rate_columns=["OriginCarrier"])

        importance = request_column_importance(
            self.a_model([0.7, 0.3]), columns, transformer, self.RAW
        )

        assert importance["OriginCarrier"] == pytest.approx(0.7)

    def test_the_ranking_is_ordered_by_weight(self):
        columns = ["OriginCongestion", "OriginCarrierDelayRate"]
        transformer = StubTransformer(delay_rate_columns=["OriginCarrier"])

        importance = request_column_importance(
            self.a_model([0.1, 0.9]), columns, transformer, self.RAW
        )

        assert list(importance) == ["OriginCarrier", "OriginCongestion"]

    def test_what_the_service_supplies_itself_is_left_out(self):
        columns = ["PrecipitationOrigin", "OriginCongestion"]
        transformer = StubTransformer()

        importance = request_column_importance(
            self.a_model([0.75, 0.25]), columns, transformer, self.RAW
        )

        assert list(importance) == ["OriginCongestion"]
        assert importance["OriginCongestion"] == pytest.approx(0.25)

    def test_a_calibrated_model_is_unwrapped_first(self):
        X = pd.DataFrame(
            {"OriginCongestion": [float(i) for i in range(20)], "Dest": [0.0, 1.0] * 10}
        )
        y = pd.Series([0, 1] * 10)
        forest = RandomForestClassifier(n_estimators=3, random_state=0).fit(X, y)
        calibrated = CalibratedClassifierCV(FrozenEstimator(forest), method="isotonic").fit(X, y)

        importance = request_column_importance(
            calibrated, list(X.columns), StubTransformer(), self.RAW
        )

        assert set(importance) <= {"OriginCongestion", "Dest"}
        assert importance

    def test_an_estimator_reporting_nothing_yields_nothing(self):
        importance = request_column_importance(
            SimpleNamespace(), ["OriginCongestion"], StubTransformer(), self.RAW
        )

        assert importance == {}

    def test_a_mismatched_count_is_refused(self):
        importance = request_column_importance(
            self.a_model([0.5, 0.5]), ["OriginCongestion"], StubTransformer(), self.RAW
        )

        assert importance == {}


class TestRequestColumnContributions:
    def a_frame(self, **columns):
        return pd.DataFrame({name: [value] for name, value in columns.items()})

    def test_a_one_hot_block_is_reported_as_its_source(self):
        X = self.a_frame(OriginAirportID_10397=1.0, OriginAirportID_11298=0.0, Distance=2.0)
        y = pd.Series([0, 1] * 4)
        wide = pd.concat([X] * 8, ignore_index=True)
        wide["OriginAirportID_10397"] = [1.0, 0.0] * 4
        wide["Distance"] = [1.0, 3.0] * 4
        model = LogisticRegression(max_iter=200).fit(wide, y)
        transformer = StubTransformer(category_keep={"OriginAirportID": {"10397", "11298"}})

        reported = request_column_contributions(model, X, transformer, "logreg", top_k=5)

        assert {item["column"] for item in reported} <= {"OriginAirportID", "Distance"}

    def test_the_category_the_flight_does_not_have_adds_nothing(self):
        X = self.a_frame(a_x=1.0, a_y=0.0)
        y = pd.Series([1, 0] * 4)
        wide = pd.DataFrame({"a_x": [1.0, 0.0] * 4, "a_y": [0.0, 1.0] * 4})
        model = LogisticRegression(max_iter=200).fit(wide, y)
        transformer = StubTransformer(category_keep={"a": {"x", "y"}})

        reported = request_column_contributions(model, X, transformer, "logreg", top_k=5)

        assert [item["column"] for item in reported] == ["a"]
        assert reported[0]["contribution"] == pytest.approx(model.coef_[0][0])

    def test_a_delay_rate_is_reported_as_the_column_it_came_from(self):
        wide = pd.DataFrame({"OriginCarrierDelayRate": [0.1, 0.9] * 4})
        y = pd.Series([0, 1] * 4)
        model = LogisticRegression(max_iter=200).fit(wide, y)
        transformer = StubTransformer(delay_rate_columns=["OriginCarrier"])

        reported = request_column_contributions(
            model, self.a_frame(OriginCarrierDelayRate=0.9), transformer, "logreg"
        )

        assert reported[0]["column"] == "OriginCarrier"

    def test_a_column_that_never_was_encoded_keeps_its_name(self):
        wide = pd.DataFrame({"PrecipitationOrigin": [0.0, 5.0] * 4})
        y = pd.Series([0, 1] * 4)
        model = LogisticRegression(max_iter=200).fit(wide, y)

        reported = request_column_contributions(
            model, self.a_frame(PrecipitationOrigin=5.0), StubTransformer(), "logreg"
        )

        assert reported[0]["column"] == "PrecipitationOrigin"

    def test_the_sign_says_which_way_it_pushed(self):
        wide = pd.DataFrame({"PrecipitationOrigin": [0.0, 5.0] * 4})
        y = pd.Series([0, 1] * 4)
        model = LogisticRegression(max_iter=200).fit(wide, y)

        wet = request_column_contributions(
            model, self.a_frame(PrecipitationOrigin=5.0), StubTransformer(), "logreg"
        )
        dry = request_column_contributions(
            model, self.a_frame(PrecipitationOrigin=-5.0), StubTransformer(), "logreg"
        )

        assert wet[0]["contribution"] > 0
        assert dry[0]["contribution"] < 0

    def test_top_k_limits_what_is_reported(self):
        wide = pd.DataFrame({"a": [0.0, 1.0] * 4, "b": [1.0, 0.0] * 4, "c": [0.5, 0.2] * 4})
        y = pd.Series([0, 1] * 4)
        model = LogisticRegression(max_iter=200).fit(wide, y)

        reported = request_column_contributions(
            model, self.a_frame(a=1.0, b=1.0, c=1.0), StubTransformer(), "logreg", top_k=2
        )

        assert len(reported) == 2

    def test_an_unreadable_model_yields_nothing(self):
        reported = request_column_contributions(
            LogisticRegression(), self.a_frame(a=1.0), StubTransformer(), "naive_bayes"
        )

        assert reported == []
