"""Tests for predicting_flight_arrival_delays.app.wrapper."""

from typing import ClassVar

import httpx
import pytest

from predicting_flight_arrival_delays.app import wrapper

SCORED = {
    "delay_probability": 0.62,
    "is_delayed": 1,
    "variant": "all",
    "threshold": 0.42,
    "weather": "ok",
}


class FakeResponse:
    def __init__(self, status_code=200, body=None, text=""):
        self.status_code = status_code
        self.is_success = 200 <= status_code < 300
        self._body = body
        self.text = text

    def json(self):
        if self._body is None:
            raise ValueError("no json")
        return self._body


@pytest.fixture
def transport(monkeypatch):
    """Answer every HTTP call from memory, recording what was sent."""

    def build(result):
        sent = []

        class FakeClient:
            def __init__(self, **kwargs):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                return False

            async def request(self, method, url, **kwargs):
                sent.append((method, url, kwargs.get("json")))
                if isinstance(result, Exception):
                    raise result
                return result

        monkeypatch.setattr(wrapper.httpx, "AsyncClient", FakeClient)
        return sent

    return build


@pytest.fixture
def api(monkeypatch):
    def build(*responses):
        remaining = list(responses)
        sent = []

        async def fake_call(method, path, **kwargs):
            sent.append((method, path, kwargs.get("json")))
            return remaining.pop(0) if remaining else {}

        monkeypatch.setattr(wrapper, "call", fake_call)
        return sent

    return build


class TestCall:
    @pytest.mark.asyncio
    async def test_a_successful_answer_comes_back_decoded(self, transport):
        transport(FakeResponse(200, {"data": {"ok": True}}))

        assert await wrapper.call("GET", "/status") == {"data": {"ok": True}}

    @pytest.mark.asyncio
    async def test_a_refusal_becomes_a_readable_error(self, transport):
        """A button has nowhere to put an exception, so failures are returned."""
        transport(FakeResponse(503, {"detail": "No model is loaded."}))

        assert "No model is loaded." in (await wrapper.call("GET", "/model/metrics"))["error"]

    @pytest.mark.asyncio
    async def test_an_unreachable_service_becomes_a_readable_error(self, transport):
        transport(httpx.ConnectError("connection refused"))

        assert "unreachable" in (await wrapper.call("GET", "/status"))["error"]

    @pytest.mark.asyncio
    async def test_a_refusal_that_is_not_json_still_reports_something(self, transport):
        transport(FakeResponse(502, None, text="Bad Gateway"))

        assert "Bad Gateway" in (await wrapper.call("GET", "/status"))["error"]


class TestRenderPrediction:
    def test_a_flight_over_the_threshold_reads_as_at_risk(self):
        assert "At risk of delay" in wrapper.render_prediction(SCORED)

    def test_a_flight_under_the_threshold_reads_as_on_time(self):
        assert "Expected on time" in wrapper.render_prediction(SCORED | {"is_delayed": 0})

    def test_the_probability_is_shown_as_a_percentage(self):
        assert "62.0%" in wrapper.render_prediction(SCORED)

    def test_a_fallback_answer_says_so(self):
        rendered = wrapper.render_prediction(
            SCORED | {"variant": "noweather", "weather": "unavailable"}
        )

        assert "fallback model" in rendered
        assert "less informed" in rendered

    def test_the_full_model_carries_no_such_warning(self):
        assert "less informed" not in wrapper.render_prediction(SCORED)


class TestPredictLookup:
    @pytest.mark.asyncio
    async def test_an_incomplete_form_is_not_sent(self, api):
        sent = api()

        answer, chart = await wrapper.predict_lookup("2026-08-26", "AA", "", 3500, "DFW", "LBB")

        assert answer == "Fill in every field."
        assert chart is None
        assert sent == []

    @pytest.mark.asyncio
    async def test_the_request_carries_what_the_api_asks_for(self, api):
        sent = api(
            {
                "data": SCORED
                | {
                    "resolved": {
                        "Origin": "DFW",
                        "Dest": "LBB",
                        "DepTimeDecimal": 7.0,
                        "Distance": 282.0,
                    }
                }
            }
        )

        await wrapper.predict_lookup("2026-08-26", "aa", "mq", 3500, "DFW", "LBB")

        method, path, payload = sent[0]
        assert (method, path) == ("POST", "/predictions/lookup?explain=true")
        assert payload["FlightDate"] == "2026-08-26"
        assert payload["MarketingCarrier"] == "aa"
        assert payload["FlightNumber"] == 3500

    @pytest.mark.asyncio
    async def test_a_date_with_a_time_on_it_is_trimmed(self, api):
        """The picker can hand back a timestamp; the API takes a date."""
        sent = api(
            {
                "data": SCORED
                | {
                    "resolved": {
                        "Origin": "DFW",
                        "Dest": "LBB",
                        "DepTimeDecimal": 7.0,
                        "Distance": 282.0,
                    }
                }
            }
        )

        await wrapper.predict_lookup("2026-08-26 00:00:00", "AA", "MQ", 3500, "DFW", "LBB")

        assert sent[0][2]["FlightDate"] == "2026-08-26"

    @pytest.mark.asyncio
    async def test_the_flight_that_was_found_is_shown_back(self, api):
        api(
            {
                "data": SCORED
                | {
                    "resolved": {
                        "Origin": "DFW",
                        "Dest": "LBB",
                        "DepTimeDecimal": 7.0,
                        "Distance": 282.0,
                    }
                }
            }
        )

        answer, _ = await wrapper.predict_lookup("2026-08-26", "AA", "MQ", 3500, "DFW", "LBB")

        assert "DFW → LBB" in answer
        assert "282 miles" in answer

    @pytest.mark.asyncio
    async def test_a_refusal_is_shown_not_swallowed(self, api):
        api({"error": "404 - No flight AA9999 on 2026-08-26."})

        answer, chart = await wrapper.predict_lookup("2026-08-26", "AA", "AA", 9999, "DFW", "LBB")

        assert "AA9999" in answer
        assert chart is None


class TestTheWaterfall:
    TERMS: ClassVar[dict] = {
        "base_value": -1.30,
        "contributions": [
            {"column": "OriginCarrier", "contribution": 0.43},
            {"column": "OriginCongestion", "contribution": 0.21},
        ],
        "other_contribution": 0.06,
        "calibration": -0.09,
        "log_odds": True,
    }

    def test_it_walks_from_a_start_to_an_answer(self):
        figure = wrapper.render_waterfall(self.TERMS, 0.314)
        labels = [t.get_text() for t in figure.axes[0].get_yticklabels()]

        assert labels[0] == "starts at"
        assert labels[-1] == "the answer"

    def test_what_did_not_make_the_list_is_still_shown(self):
        labels = [
            t.get_text()
            for t in wrapper.render_waterfall(self.TERMS, 0.314).axes[0].get_yticklabels()
        ]

        assert "everything else" in labels

    def test_calibration_is_a_step_of_its_own(self):
        labels = [
            t.get_text()
            for t in wrapper.render_waterfall(self.TERMS, 0.314).axes[0].get_yticklabels()
        ]

        assert "calibration" in labels

    def test_the_answer_bar_is_where_the_steps_add_up_to(self):
        figure = wrapper.render_waterfall(self.TERMS, 0.314)
        expected = (
            self.TERMS["base_value"]
            + sum(i["contribution"] for i in self.TERMS["contributions"])
            + self.TERMS["other_contribution"]
            + self.TERMS["calibration"]
        )

        assert figure.axes[0].patches[-1].get_width() == pytest.approx(expected)

    def test_the_probability_is_named_in_the_title(self):
        figure = wrapper.render_waterfall(self.TERMS, 0.314)

        assert "31.4%" in figure.axes[0].get_title()

    def test_nothing_to_draw_yields_no_figure(self):
        assert wrapper.render_waterfall(None, 0.3) is None
        assert wrapper.render_waterfall({"contributions": []}, 0.3) is None


class TestTheContributionsChart:
    def test_one_bar_per_reason(self):
        figure = wrapper.render_contributions(
            [
                {"column": "OriginCarrier", "contribution": 0.043},
                {"column": "PrecipitationOrigin", "contribution": -0.012},
            ]
        )

        assert len(figure.axes[0].patches) == 2

    def test_the_side_of_zero_says_which_way_it_pushed(self):
        figure = wrapper.render_contributions(
            [
                {"column": "OriginCarrier", "contribution": 0.043},
                {"column": "PrecipitationOrigin", "contribution": -0.012},
            ]
        )
        widths = [p.get_width() for p in figure.axes[0].patches]

        assert min(widths) < 0 < max(widths)

    def test_the_two_directions_are_told_apart_by_colour(self):
        figure = wrapper.render_contributions(
            [
                {"column": "a", "contribution": 0.5},
                {"column": "b", "contribution": -0.5},
            ]
        )
        colours = {p.get_facecolor() for p in figure.axes[0].patches}

        assert len(colours) == 2

    def test_nothing_to_draw_yields_no_figure(self):
        assert wrapper.render_contributions([]) is None


class TestPredictBatch:
    @pytest.mark.asyncio
    async def test_no_file_is_reported_rather_than_sent(self, api):
        sent = api()

        result = await wrapper.predict_batch(None)

        assert "error" in result.columns
        assert sent == []

    @pytest.mark.asyncio
    async def test_an_unreadable_file_is_reported(self, api, tmp_path):
        api()
        broken = tmp_path / "broken.csv"
        broken.write_bytes(b"\xff\xfe\x00not a csv")

        result = await wrapper.predict_batch(str(broken))

        assert "error" in result.columns

    @pytest.mark.asyncio
    async def test_each_scored_flight_becomes_a_row(self, api, tmp_path):
        api(
            {
                "data": {
                    "results": [
                        {"index": 0, **SCORED},
                        {
                            "index": 1,
                            **(
                                SCORED
                                | {
                                    "is_delayed": 0,
                                    "variant": "noweather",
                                    "weather": "beyond_forecast_horizon",
                                }
                            ),
                        },
                    ]
                }
            }
        )
        csv = tmp_path / "flights.csv"
        csv.write_text("FlightDate,OriginAirportID\n2026-08-26,12478\n2026-08-27,12478\n")

        result = await wrapper.predict_batch(str(csv))

        assert list(result["flight"]) == [0, 1]
        assert "At risk" in result["verdict"].iloc[0]
        assert "On time" in result["verdict"].iloc[1]

    @pytest.mark.asyncio
    async def test_a_refusal_becomes_a_single_error_row(self, api, tmp_path):
        api({"error": "422 - Flight 0 leaves out 3 columns"})
        csv = tmp_path / "flights.csv"
        csv.write_text("FlightDate\n2026-08-26\n")

        result = await wrapper.predict_batch(str(csv))

        assert "columns" in result["error"].iloc[0]


class TestModelPages:
    @pytest.mark.asyncio
    async def test_the_metrics_page_names_each_variant(self, api):
        api(
            {
                "data": {
                    "all": {
                        "run_id": "r1",
                        "operating_threshold": 0.42,
                        "metrics": {"pr_auc": 0.5},
                    }
                }
            }
        )

        page = await wrapper.get_metrics()

        assert "all" in page
        assert "0.5000" in page

    @pytest.mark.asyncio
    async def test_the_inputs_page_separates_required_from_ignored(self, api):
        api(
            {
                "data": {
                    "derived_from_served_models": True,
                    "required": ["FlightDate"],
                    "ignored": ["Month"],
                    "supplied_by_the_service": ["LeadDays"],
                }
            }
        )

        page = await wrapper.get_inputs()

        assert "FlightDate" in page
        assert "`Month`" in page
        assert "`LeadDays`" in page

    @pytest.mark.asyncio
    async def test_with_no_model_the_page_says_so_instead_of_listing(self, api):
        """The list would otherwise be captioned "feature selection dropped them",
        which is false when nothing was ever selected."""
        api(
            {
                "data": {
                    "derived_from_served_models": False,
                    "required": ["FlightDate"],
                    "ignored": ["Month"],
                    "supplied_by_the_service": ["LeadDays"],
                }
            }
        )

        page = await wrapper.get_inputs()

        assert "No model is in service" in page
        assert "Feature selection" not in page
        assert "`Month`" not in page

    @pytest.mark.asyncio
    async def test_the_reasons_are_rendered_with_their_direction(self, api):
        api(
            {
                "data": {
                    "delay_probability": 0.31,
                    "is_delayed": 0,
                    "variant": "all",
                    "threshold": 0.5,
                    "weather": "ok",
                    "explanations": [
                        {"column": "OriginCarrier", "contribution": 0.043},
                        {"column": "PrecipitationOrigin", "contribution": -0.012},
                    ],
                }
            }
        )

        page = wrapper.render_prediction((await wrapper.call("GET", "/x"))["data"])

        assert "**OriginCarrier** pushed towards a delay" in page
        assert "**PrecipitationOrigin** pushed towards arriving on time" in page

    @pytest.mark.asyncio
    async def test_missing_important_inputs_are_flagged(self, api):
        api(
            {
                "data": {
                    "delay_probability": 0.31,
                    "is_delayed": 0,
                    "variant": "all",
                    "threshold": 0.5,
                    "weather": "ok",
                    "approximated": ["OriginCongestion", "OriginCarrier"],
                }
            }
        )

        page = wrapper.render_prediction((await wrapper.call("GET", "/x"))["data"])

        assert "less accurate" in page
        assert "`OriginCongestion`" in page
        assert "`OriginCarrier`" in page

    @pytest.mark.asyncio
    async def test_a_complete_request_gets_no_warning(self, api):
        api(
            {
                "data": {
                    "delay_probability": 0.31,
                    "is_delayed": 0,
                    "variant": "all",
                    "threshold": 0.5,
                    "weather": "ok",
                    "approximated": [],
                }
            }
        )

        page = wrapper.render_prediction((await wrapper.call("GET", "/x"))["data"])

        assert "less accurate" not in page

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "page", [wrapper.get_metrics, wrapper.get_hyperparameters, wrapper.get_inputs]
    )
    async def test_every_page_stays_readable_when_the_api_refuses(self, api, page):
        api({"error": "503 - No model is loaded."})

        assert "No model is loaded." in await page()
