import os
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

# Load environment variables from .env file if it exists
load_dotenv()
SEED = 42   

# ------START Paths--------
# --Data--
PROJ_ROOT = Path(__file__).resolve().parents[1]
logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")
DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"
EXTERNAL_RAW_DATA_DIR = EXTERNAL_DATA_DIR / "raw"
# --Models--
MODELS_DIR = PROJ_ROOT / "models"
# --Reports/Figures--
REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
# ------END Paths--------


# BTS data source
BTS_BASE_URL = (
    "https://transtats.bts.gov/PREZIP/"
    "On_Time_Reporting_Carrier_On_Time_Performance_1987_present_{year}_{month}.zip"
)
YEARS_DATA = (2025, 2026) # Years data to download
MASTER_CORD_PAGE = (
    "https://transtats.bts.gov/DL_SelectFields.aspx"
    "?gnoyr_VQ=FLL&QO_fu146_anzr=N8vn6v10"
)
T_MASTER_CORD_FILE_NAME = "T_MASTER_CORD"
REQUIRED_AIRPORTS_COLUMNS = [
    "AIRPORT_ID",
    "AIRPORT",
    "LATITUDE",
    "LONGITUDE",
    "AIRPORT_IS_LATEST",
    "AIRPORT_COUNTRY_CODE_ISO",
]


# Open meteo data source
HISTORICAL_FORECAST_URL = "https://historical-forecast-api.open-meteo.com/v1/forecast"
PREVIOUS_RUNS_URL = "https://previous-runs-api.open-meteo.com/v1/forecast"
WEATHER_MODEL = "gfs_seamless"
WEATHER_COLUMNS = [
    "Temperature2m",
    "Precipitation",
    "Snowfall",
    "WindSpeed10m",
    "WindGusts10m",
    "WeatherCode",
]
WEATHER_VARS = [
    "temperature_2m",
    "precipitation",
    "snowfall",
    "wind_speed_10m",
    "wind_gusts_10m",
    "weather_code",
]
MAX_LEAD_DAYS = 5 # Upper bound on how far ahead a forecast can be requested from the weather source
FULL_LEAD_COVERAGE_START = "2024-12-01"


# BTS columns to keep
KEEP_COLUMNS = [
    "OriginAirportID", "DestAirportID",
    "Origin", "Dest",
    "Reporting_Airline", "Flight_Number_Reporting_Airline",
    "FlightDate", "CRSDepTime", "CRSArrTime", "CRSElapsedTime",
    "Distance", "DistanceGroup",
    "DayofMonth", "DayOfWeek", "Month",
    "OriginState", "DestState", "OriginCityName", "DestCityName", "Tail_Number"
]
TARGET = "IsDelayed"
# Columns needed by the pipeline but never given to the model.
SERVICE_COLUMNS = [
    "CRSDepTime",      
    "CRSArrTime",       
    "DepUtcHour",     
    "ArrUtcHour",
    "DepHour",    
    "ArrHour",
    "TailNumber",
    "Origin",
    "Dest",
]

SERVICE_COLUMNS2 = [
    "FlightDate",
]
DATE_COLUMN = "FlightDate"

#Transform
MIN_CATEGORY_COUNT = 1000
MAX_ONEHOT_CATEGORIES = 50
CORRELATION_THRESHOLD = 0.98
CATEGORICAL_ASSOCIATION_THRESHOLD = 0.98
MIN_MUTUAL_INFO = 1e-5
MI_SAMPLE_SIZE = 500_000

# ------START Training and Model Selection--------
PRODUCTION_VARIANTS = ["all", "noweather"]
ENCODING = {
    "logistic_regression": "onehot",
    "random_forest": "onehot",
    "lightgbm": "native",
}
METRICS_DIR = PROJ_ROOT / "metrics"
HYPERPARAMS_PATH = Path(__file__).resolve().parent / "modeling" / "hyperparams.yaml"
RESAMPLE_METHODS = ("none", "undersample", "oversample", "smote")
# ------END Training and Model Selection--------


# ------START MLflow tracking (DagsHub)--------
DAGSHUB_REPO_OWNER = "Beviale"
DAGSHUB_REPO_NAME = "FlightOnTime"
# ------END MLflow tracking--------


# ------START Serving--------
API_URL = os.environ.get("API_URL", "http://localhost:7860")
WINNER_MODEL_STAGE = "Champion" 
SERVED_VARIANTS = PRODUCTION_VARIANTS
DEFAULT_THRESHOLD = 0.5 
MAX_BATCH_SIZE = 200  # flights per batch request

FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
FORECAST_TIMEOUT_SECONDS = 8


# ---- Auto-lookup: flight schedules from AeroDataBox, via RapidAPI ----
AERODATABOX_HOST = "aerodatabox.p.rapidapi.com"
AERODATABOX_TIMEOUT_SECONDS = 12
DOMESTIC_COUNTRY_CODES = frozenset({"us", "pr", "vi", "gu", "mp", "as"})
# ------END Serving--------



# If tqdm is installed, configure loguru with tqdm.write
# https://github.com/Delgan/loguru/issues/135
try:
    from tqdm import tqdm

    logger.remove(0)
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
except ModuleNotFoundError:
    pass
