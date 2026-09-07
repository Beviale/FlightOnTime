# Dataset Card — FlightOnTime

## Overview

The flights this project learns from are the US Department of Transportation's own
record of every scheduled domestic flight: the **Bureau of Transportation Statistics
On-Time Performance** dataset. Around them are assembled two other sources — an airport
reference table from the same agency, and an hourly weather forecast archive — because a
flight on its own does not say where it is going or what the sky looked like when it
left.

| | |
|---|---|
| **Period covered** | 2025-01-01 to 2026-06-30 |
| **Monthly extracts** | 18 |
| **Flights after cleaning** | 10,282,102 |
| **Delayed (`IsDelayed = 1`)** | 22.25% |
| **Granularity** | one row per scheduled flight |
| **Geography** | United States, domestic carriers |

## Sources

### 1. Flights — BTS On-Time Performance

Downloaded month by month from [transtats.bts.gov](https://transtats.bts.gov/) by the
`download_bts` pipeline stage. Each monthly ZIP holds one CSV of every scheduled domestic
flight, roughly 600,000 rows.

**Licence:** US federal government work, public domain.

### 2. Airports — BTS T_MASTER_CORD

The master coordinate table, which gives every airport id its latitude, longitude and
country. The `build_airports` stage keeps the entries marked as current and derives the
time zone from the coordinates, which is what makes a local scheduled time convertible to
UTC — and therefore joinable to a weather forecast.

A second table, built by `build_airport_identity`, maps each **IATA code** to the airport
**ids** that have used it. Codes are reassigned: two airports can share `AUS` across the
period, and keying anything on the code would blend them. Where a code has been reused,
the most recent flight's naming is kept and a warning is logged.

**Licence:** US federal government work, public domain.

### 3. Weather — Open-Meteo

Hourly forecasts for every airport and date, fetched by `download_weather` from
[Open-Meteo](https://open-meteo.com/) using the `gfs_seamless` model.

The important detail is **which** forecast is fetched: not the weather that occurred, but
the forecast that *would have been available* at the lead time a passenger had. Training
on observed weather would leak information the model can never have at prediction time,
and would flatter every metric in this repository.

Six variables are kept, at origin and at destination:

`Temperature2m`, `Precipitation`, `Snowfall`, `WindSpeed10m`, `WindGusts10m`, `WeatherCode`

**Licence:** CC-BY 4.0, free for non-commercial use.

### 4. Schedules — AeroDataBox *(serving only)*

Used at inference time, never in training. When a caller names a flight rather than
describing it, [AeroDataBox](https://aerodatabox.com/) via RapidAPI supplies the
timetable, the aircraft rotation and the movement counts that become the congestion
features.

**Licence:** commercial, metered API.

## Target

**`IsDelayed`** — derived from the BTS `ArrDel15` flag: 1 when the flight arrived fifteen
or more minutes after its scheduled time, 0 otherwise.

Fifteen minutes is not an arbitrary cut. It is the threshold the BTS itself uses to
publish on-time performance, so the label matches the number airlines are held to
publicly.

### What is removed, and why

`load_and_clean` drops three kinds of row:

| Dropped | Reason |
|---------|--------|
| Cancelled flights | never arrived, so there is no arrival delay to predict |
| Diverted flights | arrived somewhere else; the label would describe a different flight |
| `ArrDel15` missing | no outcome recorded |

The three overlap almost exactly: measured across all 18 extracts, `ArrDel15` is null **if
and only if** the flight was cancelled or diverted — zero unexplained nulls in 3.6 million
sampled rows.

A fourth filter was added later, for a defect described below.

## Features

Nineteen columns are read from the raw extracts (`KEEP_COLUMNS`), and the pipeline derives
the rest.

### From the timetable

`FlightDate`, `Month`, `DayOfWeek`, `CRSDepTime`, `CRSArrTime`, `CRSElapsedTime`,
`Distance`, `DistanceGroup`

Scheduled times arrive as HHMM integers and are converted to decimal hours, then encoded
as sine and cosine pairs — 23:00 and 01:00 are two hours apart, and a raw hour number
would place them twenty-two apart.

### Identity

`OriginAirportID`, `DestAirportID`, `Origin`, `Dest`, `OriginState`, `DestState`,
`OriginCityName`, `DestCityName`, `Reporting_Airline`,
`Flight_Number_Reporting_Airline`, `Tail_Number`

### Derived

| Feature | What it is |
|---------|-----------|
| `OriginCarrier`, `DestCarrier` | airport id joined to airline code, e.g. `11298AA` — always replaced by that pairing's historical delay rate |
| `OriginCongestion`, `DestCongestion` | flights sharing the airport and the scheduled hour |
| `AircraftDailyLegs`, `LegPosition` | how many flights this aircraft flies that day, and which one this is |
| `ScheduledTurnaround` | minutes since the aircraft's previous leg |
| `IsHoliday`, `DaysToNearestHoliday` | US federal holidays |
| `DepUtcHour`, `ArrUtcHour` | the join keys to the weather archive |

**Why the delay rates are keyed on the airport id and not the code.** Measured directly:
two airports sharing the code `AUS` produce a blended rate of 0.5, while keying on
`OriginAirportID` separates them into 0.0 and 1.0. The code is ambiguous; the id is not.

**Why the bare flight number is not a feature.** 92.3% of flight numbers are used by more
than one airline, 4.77 on average. As a category it identifies nothing.

## Splits

Walk-forward, never random. Three folds, each training on the past and measured on the
future, with train, validation and test contiguous and no gaps.

| Fold | Train | Validation | Test |
|------|-------|------------|------|
| 1 | 2025-01-01 → 2025-05-28 (2.73M) | 2025-05-29 → 2025-06-30 (0.66M) | 2025-07-01 → 2025-09-30 (1.76M) |
| 2 | 2025-01-01 → 2025-08-07 (4.14M) | 2025-08-08 → 2025-09-30 (1.01M) | 2025-10-01 → 2025-12-31 (1.73M) |
| 3 | 2025-01-01 → 2025-10-19 (5.52M) | 2025-10-20 → 2025-12-31 (1.36M) | 2026-01-01 → 2026-06-01 (2.79M) |

A random split would let the model learn from August to predict March. No deployed model
can do that, and a metric obtained that way is not a forecast of anything.

The validation window is where calibration is fitted and the operating threshold chosen,
so neither touches the test period.

**The base rate moves between folds** — 27.6%, 18.9%, 23.5% on validation — which is why
PR-AUC differs across folds even when the model does not. It tracks the base rate almost
exactly.

## Variants

Three feature sets are trained; two are served.

| Variant | Difference | Role |
|---------|-----------|------|
| **all** | every feature | served when the forecast resolves |
| **noweather** | weather columns removed | served when it does not, and a measure of what weather is worth |
| **nocarrier** | carrier delay rates removed | ablation only, never deployed |

## Known data quality issues

Validated on every monthly extract by the Great Expectations suite in
`tests/test_flight_data/raw_test.py`. Two findings are worth recording.

### Impossible scheduled block times

Three months carry a single corrupt row each:

| Month | `CRSElapsedTime` | Cancelled? |
|-------|------------------|-----------|
| 2025_03 | 1510 minutes (over 25 hours) | yes |
| 2025_11 | **−60 minutes** | **no** |
| 2025_12 | −61 minutes | yes |

The negative values look like the supplier's own arithmetic across a clock change. The
2025_11 row is the one that matters: it is not cancelled, so it survived the filter, and
`add_utc_features` adds that block time to the departure — placing the arrival an hour
*before* the departure and reading the wrong hour's weather.

`load_and_clean` now drops any row whose block time is outside 1 to 1440 minutes, and logs
what it dropped. The raw suite tolerates a handful of such rows (`mostly=0.9999`) because
they come from the supplier; the processed suite tolerates none, because by then they
should be gone.

### Cancellation rate varies by month

`ArrDel15` nulls range from 0.62% (September 2025) to 6.07% (January 2026). An earlier
expectation asserted at most 5% nulls, which failed in January — but that is a statement
about winter weather, not about data quality. The expectation is now conditioned on
`Cancelled == 0 and Diverted == 0`, which is what it always meant.

## Versioning

Every artifact is tracked with **DVC**, with the remote on
[DagsHub](https://dagshub.com/Beviale/FlightOnTime). `dvc.lock` pins the exact hash of
every stage output, so a commit identifies both the code and the data it ran on.

## Ethical considerations

The dataset contains no personal data. Flights are identified by number and aircraft
registration, never by passenger.

It does, however, describe **airline performance**, and a model built on it necessarily
learns that some carriers are late more often than others. `OriginCarrier` and
`DestCarrier` are among the features the model leans on most. This is factual rather than
prejudicial — it is what the public record says — but it means predictions carry a
judgement about specific airlines, and the `nocarrier` variant exists partly to measure
how much of the model's skill rests on that judgement. The answer is about 0.6 points of
ROC-AUC.

See [Risk_Classification.md](Risk_Classification.md) for the assessment under the AI Act.
