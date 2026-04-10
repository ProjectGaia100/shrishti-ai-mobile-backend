import hmac
import logging
import os
import time
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
DATASET_SYNC_STATUS: Dict[str, Any] = {}


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


def _setup_hf_dataset_paths() -> Dict[str, Any]:
    started = time.perf_counter()
    repo_id = (config.DATASET_REPO_ID or "").strip()
    status: Dict[str, Any] = {
        "enabled": bool(repo_id),
        "repo_id": repo_id,
        "token_configured": bool((config.HF_TOKEN or "").strip()),
        "local_dir": "",
        "snapshot_dir": "",
        "success": False,
        "assigned_raster_paths": 0,
        "total_expected_rasters": 9,
        "missing_rasters": [],
        "state_data_root": "",
        "state_data_exists": False,
        "raster_files": {},
        "message": "",
    }

    if not repo_id:
        logger.info("DATASET_REPO_ID is not set, using existing raster path configuration")
        status["message"] = "DATASET_REPO_ID not set"
        status["duration_seconds"] = round(time.perf_counter() - started, 3)
        return status

    if config.DATASET_LOCAL_DIR:
        local_dir = Path(config.DATASET_LOCAL_DIR).resolve()
    else:
        local_dir = Path("/tmp/shrishti_mobile_data") if Path("/tmp").is_dir() else Path(__file__).resolve().parent / "hf_data"

    status["local_dir"] = str(local_dir)
    logger.info("Syncing dataset %s into %s", repo_id, local_dir)
    logger.info("Dataset sync filters: %s", ["final_lookup_tables/*.tif", "state_data/**"])

    try:
        dataset_dir = snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            local_dir=str(local_dir),
            allow_patterns=["final_lookup_tables/*.tif", "state_data/**"],
            token=(config.HF_TOKEN or "").strip() or None,
            local_dir_use_symlinks=False,
        )
    except Exception as exc:
        logger.warning("Dataset sync failed, continuing with existing raster paths: %s", exc)
        status["message"] = f"Dataset sync failed: {exc}"
        status["duration_seconds"] = round(time.perf_counter() - started, 3)
        return status

    dataset_root = Path(dataset_dir)
    raster_root = dataset_root / "final_lookup_tables"
    state_root = dataset_root / "state_data"
    status["snapshot_dir"] = str(dataset_root)
    status["state_data_root"] = str(state_root)

    raster_env_map = {
        "RASTER_SOIL_PATH": "soil_type.tif",
        "RASTER_ELEVATION_PATH": "elevation.tif",
        "RASTER_POPULATION_PATH": "population_density.tif",
        "RASTER_LANDCOVER_PATH": "land_cover.tif",
        "RASTER_NDVI_PATH": "ndvi.tif",
        "RASTER_PRECIP_PATH": "annual_precip.tif",
        "RASTER_TEMP_PATH": "mean_annual_temp.tif",
        "RASTER_WIND_PATH": "wind_speed.tif",
        "RASTER_IMPERVIOUS_PATH": "impervious_surface.tif",
    }

    assigned = 0
    missing = []
    for env_key, filename in raster_env_map.items():
        raster_path = raster_root / filename
        exists = raster_path.exists()
        size_mb = round(raster_path.stat().st_size / (1024 * 1024), 2) if exists else 0
        status["raster_files"][filename] = {
            "path": str(raster_path),
            "exists": exists,
            "size_mb": size_mb,
            "env_key": env_key,
        }

        logger.info(
            "Dataset raster check: %s -> %s | exists=%s | size_mb=%s",
            env_key,
            raster_path,
            exists,
            size_mb,
        )

        if raster_path.exists():
            os.environ[env_key] = str(raster_path)
            assigned += 1
            logger.info("Assigned %s=%s", env_key, raster_path)
        else:
            missing.append(filename)

    if state_root.exists():
        os.environ["STATE_DATA_ROOT"] = str(state_root)
        status["state_data_exists"] = True
        try:
            state_subdirs = sorted([p.name for p in state_root.iterdir() if p.is_dir()])
        except Exception:
            state_subdirs = []
        status["state_subdirs"] = state_subdirs
        logger.info(
            "Assigned STATE_DATA_ROOT=%s | subdirs=%s",
            state_root,
            len(state_subdirs),
        )
    else:
        logger.warning("state_data directory not found under dataset snapshot: %s", state_root)

    status["success"] = True
    status["assigned_raster_paths"] = assigned
    status["missing_rasters"] = missing
    status["message"] = "Dataset sync completed"
    status["duration_seconds"] = round(time.perf_counter() - started, 3)

    logger.info("Dataset sync complete, configured %s raster paths from %s", assigned, raster_root)
    if missing:
        logger.warning("Missing raster files after dataset sync: %s", ", ".join(missing))

    return status


def _load_hazardguard_service():
    global DATASET_SYNC_STATUS
    model_root = _resolve_model_root()
    _ensure_models_downloaded(model_root)
    DATASET_SYNC_STATUS = _setup_hf_dataset_paths()
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
                "diagnostics": "/api/mobile/diagnostics",
                "scheduler_run": "/api/scheduler/run-mobile-alerts",
            },
        }
    )


@app.get("/health")
def health():
    configured = bool(config.SUPABASE_URL and config.SUPABASE_SERVICE_ROLE_KEY)
    service_status = hazardguard_service.get_service_status()
    return jsonify(
        {
            "success": True,
            "service": "mobile-backend-hf2",
            "configured": configured,
            "model_predict_url_set": bool(config.MOBILE_HF_PREDICTION_URL),
            "dataset_sync": DATASET_SYNC_STATUS,
            "hazardguard": {
                "service_status": service_status.get("service_status"),
                "model_loaded": service_status.get("model_loaded"),
                "total_requests": (service_status.get("statistics") or {}).get("total_requests", 0),
                "failed_predictions": (service_status.get("statistics") or {}).get("failed_predictions", 0),
                "weather_failures": (service_status.get("statistics") or {}).get("weather_fetch_failures", 0),
                "feature_failures": (service_status.get("statistics") or {}).get("feature_engineering_failures", 0),
                "raster_failures": (service_status.get("statistics") or {}).get("raster_fetch_failures", 0),
            },
        }
    )


@app.get("/api/mobile/diagnostics")
def mobile_diagnostics():
    if not _authorize_scheduler():
        return _json_error("Invalid scheduler secret", 401)

    return jsonify(
        {
            "success": True,
            "service": "mobile-backend-hf2",
            "dataset_sync": DATASET_SYNC_STATUS,
            "hazardguard_status": hazardguard_service.get_service_status(),
            "scheduler_runtime": scheduler_service.get_runtime_status(),
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

    try:
        outcome = scheduler_service.predict_for_location_with_retry(location)
    except Exception as exc:
        logger.exception("/api/mobile/predict failed with unhandled exception")
        return jsonify({"success": False, "error": f"Unhandled predict exception: {exc}"}), 500

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
