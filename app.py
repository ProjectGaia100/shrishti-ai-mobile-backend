import hmac
import logging
import os
from pathlib import Path
from typing import Any, Dict

from flask import Flask, jsonify, request
from flask_cors import CORS
from huggingface_hub import snapshot_download

from config import AppConfig
from mobile_alert_scheduler_service import MobileAlertSchedulerService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

config = AppConfig.from_env()


def _resolve_model_root() -> Path:
    if config.MODEL_ROOT_PATH:
        return Path(config.MODEL_ROOT_PATH).resolve()
    return Path(__file__).resolve().parent / "models"


def _ensure_models_downloaded(model_root: Path):
    required_file = model_root / "hazardguard" / "normal_vs_disaster" / "normal_vs_disaster_xgboost_model.pkl"
    if required_file.exists():
        logger.info("HazardGuard models already present at %s", model_root)
        return

    if config.SKIP_MODEL_DOWNLOAD:
        logger.warning("SKIP_MODEL_DOWNLOAD=true and models not found at %s", model_root)
        return

    if not config.HF_TOKEN:
        raise RuntimeError("HF_TOKEN is required to download private HazardGuard models")

    logger.info("Downloading models from %s into %s", config.MODEL_REPO_ID, model_root)
    model_root.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=config.MODEL_REPO_ID,
        repo_type="model",
        local_dir=str(model_root),
        token=config.HF_TOKEN,
        allow_patterns=["hazardguard/**"],
        ignore_patterns=["*.git*", ".gitattributes"],
    )


def _load_hazardguard_service():
    model_root = _resolve_model_root()
    _ensure_models_downloaded(model_root)
    os.environ["MODEL_ROOT_PATH"] = str(model_root)

    from services.weather_service import NASAPowerService
    from services.feature_engineering_service import FeatureEngineeringService
    from services.raster_data_service import RasterDataService
    from services.hazardguard_prediction_service import HazardGuardPredictionService
    from raster_config import get_raster_config

    weather_service = NASAPowerService()
    feature_service = FeatureEngineeringService()
    raster_service = RasterDataService(get_raster_config().get_config())

    hazardguard_service = HazardGuardPredictionService(
        weather_service=weather_service,
        feature_service=feature_service,
        raster_service=raster_service,
    )
    success, message = hazardguard_service.initialize_service()
    if not success:
        raise RuntimeError(f"Failed to initialize local HazardGuard service: {message}")

    logger.info("Local HazardGuard service initialized successfully")
    return hazardguard_service


hazardguard_service = _load_hazardguard_service()

app = Flask(__name__)
app.config.update(config.__dict__)

CORS(app, origins=config.ALLOWED_ORIGINS)

scheduler_service = MobileAlertSchedulerService(
    supabase_url=config.SUPABASE_URL,
    supabase_service_role_key=config.SUPABASE_SERVICE_ROLE_KEY,
    model_predict_url=config.MOBILE_HF_PREDICTION_URL,
    hazardguard_service=hazardguard_service,
    model_api_token=config.MOBILE_HF_API_TOKEN,
    risk_threshold=config.MOBILE_ALERT_RISK_THRESHOLD,
    forecast_hours=config.MOBILE_ALERT_FORECAST_HOURS,
    batch_size=config.MOBILE_ALERT_BATCH_SIZE,
    request_timeout_seconds=config.MOBILE_HF_TIMEOUT_SECONDS,
    retry_backoff_seconds=config.MOBILE_HF_BACKOFF_SECONDS,
)


def _json_error(message: str, status: int):
    return jsonify({"success": False, "message": message}), status


def _is_valid_secret(expected: str, provided: str) -> bool:
    if not expected:
        return False
    return hmac.compare_digest(str(expected), str(provided))


def _authorize_mobile_client() -> bool:
    expected = config.MOBILE_CLIENT_API_KEY
    if not expected:
        return True
    provided = request.headers.get("X-Mobile-Api-Key", "")
    return _is_valid_secret(expected, provided)


def _authorize_scheduler() -> bool:
    provided = request.headers.get("X-Scheduler-Secret", "")
    return _is_valid_secret(config.SCHEDULER_SECRET, provided)


@app.get("/")
def root():
    return jsonify(
        {
            "success": True,
            "service": "mobile-backend-hf2",
            "endpoints": {
                "health": "/health",
                "predict": "/api/mobile/predict",
                "scheduler_run": "/api/scheduler/run-mobile-alerts",
            },
        }
    )


@app.get("/health")
def health():
    configured = bool(config.SUPABASE_URL and config.SUPABASE_SERVICE_ROLE_KEY)
    return jsonify(
        {
            "success": True,
            "service": "mobile-backend-hf2",
            "configured": configured,
            "model_predict_url_set": bool(config.MOBILE_HF_PREDICTION_URL),
        }
    )


@app.post("/api/mobile/predict")
def mobile_predict():
    if not _authorize_mobile_client():
        return _json_error("Unauthorized mobile client", 401)

    body: Dict[str, Any] = request.get_json(silent=True) or {}
    lat = body.get("latitude")
    lon = body.get("longitude")

    if lat is None or lon is None:
        return _json_error("latitude and longitude are required", 400)

    location = {
        "id": body.get("location_id", "on_demand"),
        "user_id": body.get("user_id", "on_demand"),
        "city": body.get("city", ""),
        "country": body.get("country", ""),
        "lat": lat,
        "lon": lon,
    }

    outcome = scheduler_service.predict_for_location_with_retry(location)
    if not outcome.success:
        return jsonify({"success": False, "error": outcome.error}), 502

    return jsonify(
        {
            "success": True,
            "data": {
                "risk_score": outcome.risk_score,
                "confidence": outcome.confidence,
                "disaster_type": outcome.disaster_type,
                "prediction_timestamp": outcome.prediction_timestamp,
            },
        }
    )


@app.post("/api/scheduler/run-mobile-alerts")
def run_scheduler():
    if not _authorize_scheduler():
        return _json_error("Invalid scheduler secret", 401)

    body = request.get_json(silent=True) or {}
    dry_run = bool(body.get("dry_run", False))

    result = scheduler_service.run_cycle(dry_run=dry_run)
    if result.get("status") == "skipped":
        return jsonify(result), 202
    if not result.get("success"):
        return jsonify(result), 500
    return jsonify(result), 200


if __name__ == "__main__":
    app.run(host=config.FLASK_HOST, port=config.FLASK_PORT, debug=config.FLASK_DEBUG)
