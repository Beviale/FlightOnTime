import os
import random
from datetime import date, timedelta

from locust import HttpUser, between, task

AIRPORTS = [
    (10397, "ATL", "Atlanta, GA", "GA"),
    (11057, "CLT", "Charlotte, NC", "NC"),
    (11292, "DEN", "Denver, CO", "CO"),
    (11298, "DFW", "Dallas/Fort Worth, TX", "TX"),
    (12478, "JFK", "New York, NY", "NY"),
    (12889, "LAS", "Las Vegas, NV", "NV"),
    (12892, "LAX", "Los Angeles, CA", "CA"),
    (13204, "MCO", "Orlando, FL", "FL"),
    (13303, "MIA", "Miami, FL", "FL"),
    (13930, "ORD", "Chicago, IL", "IL"),
    (14107, "PHX", "Phoenix, AZ", "AZ"),
    (14747, "SEA", "Seattle, WA", "WA"),
]

CARRIERS = ["AA", "DL", "UA", "WN", "AS", "B6", "NK", "F9"]


LEAD_DAYS_MIN = int(os.environ.get("LOCUST_LEAD_DAYS_MIN", "10"))
LEAD_DAYS_MAX = int(os.environ.get("LOCUST_LEAD_DAYS_MAX", "40"))


def a_flight() -> dict:
    """One complete scheduled flight, plausible in every field."""
    (origin_id, origin, origin_city, origin_state), (dest_id, dest, dest_city, dest_state) = (
        random.sample(AIRPORTS, 2)
    )
    carrier = random.choice(CARRIERS)

    flight_date = date.today() + timedelta(days=random.randint(LEAD_DAYS_MIN, LEAD_DAYS_MAX))
    departure = round(random.uniform(5.0, 22.0), 2)
    block_minutes = random.randint(60, 380)
    arrival = round((departure + block_minutes / 60) % 24, 2)

    distance = round(block_minutes * random.uniform(6.5, 8.5))

    return {
        "FlightDate": flight_date.isoformat(),
        "OriginAirportID": origin_id,
        "DestAirportID": dest_id,
        "DepTimeDecimal": departure,
        "ArrTimeDecimal": arrival,
        "CRSElapsedTime": float(block_minutes),
        "Month": flight_date.month,
        "DayOfWeek": flight_date.weekday() + 1,
        "IsHoliday": 0,
        "DaysToNearestHoliday": random.randint(-60, 60),
        "ReportingAirline": carrier,
        "FlightNumberReportingAirline": random.randint(1, 6000),
        "Origin": origin,
        "OriginCityName": origin_city,
        "OriginState": origin_state,
        "Dest": dest,
        "DestCityName": dest_city,
        "DestState": dest_state,
        "OriginCarrier": f"{origin_id}{carrier}",
        "DestCarrier": f"{dest_id}{carrier}",
        "Distance": float(distance),
        "DistanceGroup": min(11, distance // 250 + 1),
        "OriginCongestion": random.randint(1, 60),
        "DestCongestion": random.randint(1, 60),
        "AircraftDailyLegs": random.randint(1, 6),
        "LegPosition": random.randint(1, 4),
        "ScheduledTurnaround": float(random.randint(30, 240)),
    }


class Traveller(HttpUser):
    wait_time = between(1, 3)

    @task(5)
    def score_one_flight(self):
        self.client.post("/predictions", json=a_flight(), name="/predictions")

    @task(2)
    def score_a_batch(self):
        payload = [a_flight() for _ in range(random.randint(5, 20))]
        self.client.post("/batch-predictions", json=payload, name="/batch-predictions")

    @task(2)
    def ask_why(self):
        self.client.post("/explanations", json=a_flight(), name="/explanations")

    @task(3)
    def open_the_interface(self):
        self.client.get("/", name="/ (Gradio)")

    @task(2)
    def read_the_metrics(self):
        self.client.get("/model/metrics")

    @task(1)
    def read_the_hyperparameters(self):
        self.client.get("/model/hyperparameters")

    @task(1)
    def read_the_inputs(self):
        self.client.get("/model/inputs")
