import os
from dataclasses import dataclass
from typing import List

from dotenv import load_dotenv

load_dotenv()


def _parse_int_list(raw: str, default: List[int]) -> List[int]:
    values: List[int] = []
    for part in (raw or "").split(","):
        part = part.strip()
        if not part:
            continue
        try:
            values.append(max(int(part), 1))
        except ValueError:
            continue
    return values or default


@dataclass
class AppConfig:
    FLASK_ENV: str
    FLASK_DEBUG: bool
    FLASK_HOST: str
    FLASK_PORT: int
    ALLOWED_ORIGINS: List[str]

    SCHEDULER_SECRET: str
    MOBILE_CLIENT_API_KEY: str

    SUPABASE_URL: str
    SUPABASE_SERVICE_ROLE_KEY: str

    MODEL_REPO_ID: str
    MODEL_ROOT_PATH: str
    HF_TOKEN: str
    SKIP_MODEL_DOWNLOAD: bool
    DATASET_REPO_ID: str
    DATASET_LOCAL_DIR: str

    MOBILE_HF_PREDICTION_URL: str
    MOBILE_HF_API_TOKEN: str
    MOBILE_HF_TIMEOUT_SECONDS: int
    MOBILE_HF_BACKOFF_SECONDS: List[int]

    MOBILE_ALERT_RISK_THRESHOLD: float
    MOBILE_ALERT_FORECAST_HOURS: int
    MOBILE_ALERT_BATCH_SIZE: int

    @staticmethod
    def from_env() -> "AppConfig":
        return AppConfig(
            FLASK_ENV=os.getenv("FLASK_ENV", "production"),
            FLASK_DEBUG=os.getenv("FLASK_DEBUG", "False").lower() == "true",
            FLASK_HOST=os.getenv("FLASK_HOST", "0.0.0.0"),
            FLASK_PORT=int(os.getenv("FLASK_PORT", 7860)),
            ALLOWED_ORIGINS=[x.strip() for x in os.getenv("ALLOWED_ORIGINS", "*").split(",") if x.strip()],
            SCHEDULER_SECRET=os.getenv("SCHEDULER_SECRET", ""),
            MOBILE_CLIENT_API_KEY=os.getenv("MOBILE_CLIENT_API_KEY", ""),
            SUPABASE_URL=os.getenv("SUPABASE_URL", ""),
            SUPABASE_SERVICE_ROLE_KEY=os.getenv("SUPABASE_SERVICE_ROLE_KEY", ""),
            MODEL_REPO_ID=os.getenv("MODEL_REPO_ID", "projectgaia/ShrishtiAI-models"),
            MODEL_ROOT_PATH=os.getenv("MODEL_ROOT_PATH", ""),
            HF_TOKEN=os.getenv("HF_TOKEN", ""),
            SKIP_MODEL_DOWNLOAD=os.getenv("SKIP_MODEL_DOWNLOAD", "false").lower() == "true",
            DATASET_REPO_ID=os.getenv("DATASET_REPO_ID", ""),
            DATASET_LOCAL_DIR=os.getenv("DATASET_LOCAL_DIR", ""),
            MOBILE_HF_PREDICTION_URL=os.getenv("MOBILE_HF_PREDICTION_URL", ""),
            MOBILE_HF_API_TOKEN=os.getenv("MOBILE_HF_API_TOKEN", ""),
            MOBILE_HF_TIMEOUT_SECONDS=int(os.getenv("MOBILE_HF_TIMEOUT_SECONDS", 120)),
            MOBILE_HF_BACKOFF_SECONDS=_parse_int_list(
                os.getenv("MOBILE_HF_BACKOFF_SECONDS", "10,25,45,90"),
                [10, 25, 45, 90],
            ),
            MOBILE_ALERT_RISK_THRESHOLD=float(os.getenv("MOBILE_ALERT_RISK_THRESHOLD", 0.65)),
            MOBILE_ALERT_FORECAST_HOURS=int(os.getenv("MOBILE_ALERT_FORECAST_HOURS", 24)),
            MOBILE_ALERT_BATCH_SIZE=int(os.getenv("MOBILE_ALERT_BATCH_SIZE", 5)),
        )
