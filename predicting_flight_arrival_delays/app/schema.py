"""Request models for the serving API, one per way in.

FlightRequest is manual entry: the caller supplies every feature the served models
read except the weather. The service adds only  what it alone can know - how far ahead
the request is being made, and the forecast at both ends of the route.

FlightLookupRequest is auto-lookup: the caller names a flight instead - the date, the
number, the two airports and the two carrier codes - and app.enrichment.lookup
recovers the rest from the schedule service, handing back a FlightRequest. Every one
of its six fields is required, because six is already the least that identifies a
single flight.

What follows describes manual entry, where the question of what to ask for arises.

Which of these fields are actually required is not fixed here. It is read from the
loaded bundles at startup, because feature selection decides it: a column both
released models dropped is not worth asking for. See app.inputs. The fields the
forecast lookup needs are required unconditionally, since without them there is
nowhere and no hour to ask about; the rest are optional at this level and checked
against the loaded models when the request arrives.

Fields that are functions of other fields - the calendar breakdown of the flight date,
the distance group, the airport-carrier combinations - are sent by the caller like any
other, and taken as sent. Nothing here recomputes them or checks them against the
fields they derive from.
"""

from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import BaseModel, Field, model_validator

# BTS groups distance into 250-mile buckets, everything from 2500 miles up in the last.
MAX_DISTANCE_GROUP = 11

CODE_FIELDS = [
    "ReportingAirline",
    "Origin",
    "Dest",
    "OriginState",
    "DestState",
    "OriginCarrier",
    "DestCarrier",
]


class FlightRequest(BaseModel):
    """One scheduled flight, described in full except for the weather."""

    # --- Needed to reach the forecast, whatever the models make of them
    FlightDate: date
    OriginAirportID: int = Field(gt=0, examples=[12478])
    DestAirportID: int = Field(gt=0, examples=[12892])
    DepTimeDecimal: float = Field(ge=0, lt=24, description="Local hour at origin, 14:30 is 14.5")
    CRSElapsedTime: float = Field(gt=0, description="Scheduled block time in minutes")

    # --- Required only insofar as the served models read them; see app.inputs
    Month: int | None = Field(default=None, ge=1, le=12)
    DayOfWeek: int | None = Field(default=None, ge=1, le=7, description="Monday is 1")
    IsHoliday: Literal[0, 1] | None = None
    DaysToNearestHoliday: int | None = None

    ReportingAirline: str | None = Field(default=None, min_length=1, max_length=3, examples=["AA"])
    FlightNumberReportingAirline: int | None = Field(default=None, gt=0, examples=[100])
    Origin: str | None = Field(default=None, min_length=3, max_length=3, examples=["JFK"])
    OriginCityName: str | None = Field(default=None, examples=["New York, NY"])
    OriginState: str | None = Field(default=None, min_length=2, max_length=2, examples=["NY"])
    Dest: str | None = Field(default=None, min_length=3, max_length=3, examples=["LAX"])
    DestCityName: str | None = Field(default=None, examples=["Los Angeles, CA"])
    DestState: str | None = Field(default=None, min_length=2, max_length=2, examples=["CA"])
    # The airport half is the id, matching OriginAirportID/DestAirportID above.
    OriginCarrier: str | None = Field(default=None, examples=["12478AA"])
    DestCarrier: str | None = Field(default=None, examples=["12892AA"])

    ArrTimeDecimal: float | None = Field(default=None, ge=0, lt=24)
    Distance: float | None = Field(default=None, gt=0, description="Miles")
    DistanceGroup: int | None = Field(default=None, ge=1, le=MAX_DISTANCE_GROUP)

    OriginCongestion: int | None = Field(default=None, ge=0)
    DestCongestion: int | None = Field(default=None, ge=0)
    AircraftDailyLegs: int | None = Field(default=None, ge=1)
    LegPosition: int | None = Field(default=None, ge=1)
    ScheduledTurnaround: float | None = Field(
        default=None, description="Minutes since the aircraft's previous leg"
    )

    @model_validator(mode="before")
    @classmethod
    def normalise_codes(cls, data: object) -> object:
        """Uppercase the codes, which the training vocabulary stores uppercase."""
        if not isinstance(data, dict):
            return data

        return data | {
            field: data[field].strip().upper()
            for field in CODE_FIELDS
            if isinstance(data.get(field), str)
        }

    def supplied(self) -> set[str]:
        """Which fields the caller actually sent.

        Returns:
            The names of the fields present in the request body.
        """
        return set(self.model_dump(exclude_unset=True))


class FlightLookupRequest(BaseModel):
    """One scheduled flight, named the way a ticket names it.

    Six fields, and every one of them earns its place:

        the two airports
        MarketingCarrier   the code the flight is sold under.
        ReportingAirline   the code of the airline actually operating, which is what
                           BTS records and what the model was trained on.
    """

    FlightDate: date
    MarketingCarrier: str = Field(
        min_length=1,
        max_length=3,
        examples=["AA"],
        description="The code the flight is sold under",
    )
    ReportingAirline: str = Field(
        min_length=1,
        max_length=3,
        examples=["MQ"],
        description="The code of the airline operating it",
    )
    FlightNumber: int = Field(gt=0, examples=[3500])
    Origin: str = Field(min_length=3, max_length=3, examples=["DFW"])
    Dest: str = Field(min_length=3, max_length=3, examples=["LBB"])

    @model_validator(mode="before")
    @classmethod
    def normalise_codes(cls, data: object) -> object:
        """Uppercase the codes, which both the service and the model expect uppercase."""
        if not isinstance(data, dict):
            return data

        fields = ["MarketingCarrier", "ReportingAirline", "Origin", "Dest"]
        return data | {
            field: data[field].strip().upper()
            for field in fields
            if isinstance(data.get(field), str)
        }
