from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import requests
from supabase import Client, create_client

logger = logging.getLogger(__name__)
_ALLOWED_DISASTER_TYPES = {"FLOOD", "DROUGHT", "STORM", "LANDSLIDE"}


@dataclass
class PredictionOutcome:
    success: bool
    risk_score: float = 0.0
    confidence: float = 0.0
    disaster_type: str = "STORM"
    prediction_timestamp: str = ""
    raw: Optional[Dict[str, Any]] = None
    error: str = ""


class MobileAlertSchedulerService:
    def __init__(
        self,
        supabase_url: str,
        supabase_service_role_key: str,
        model_predict_url: str,
        hazardguard_service: Any = None,
        model_api_token: str = "",
        risk_threshold: float = 0.65,
        forecast_hours: int = 24,
        batch_size: int = 5,
        request_timeout_seconds: int = 120,
        retry_backoff_seconds: Optional[List[int]] = None,
    ):
        if not supabase_url or not supabase_service_role_key:
            raise ValueError("SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY are required")

        self.supabase: Client = create_client(supabase_url, supabase_service_role_key)
        self.model_predict_url = model_predict_url
        self.hazardguard_service = hazardguard_service
        self.model_api_token = model_api_token
        self.risk_threshold = float(risk_threshold)
        self.forecast_hours = int(forecast_hours)
        self.batch_size = max(int(batch_size), 1)
        self.request_timeout_seconds = max(int(request_timeout_seconds), 10)
        self.retry_backoff_seconds = retry_backoff_seconds or [10, 25, 45, 90]
        self.last_prediction_error = ""
        self.last_prediction_error_at = ""

        self._run_lock = threading.Lock()

    def _is_non_retryable_error(self, error_message: str) -> bool:
        msg = (error_message or "").lower()
        if not msg:
            return False
        non_retryable_markers = [
            "service not initialized",
            "location lat/lon missing",
            "invalid coordinates",
            "raster data collection failed",
            "feature engineering failed",
            "weather data collection failed",
            "missing risk score",
            "response missing disaster probability",
            "model endpoint http 400",
            "model endpoint http 401",
            "model endpoint http 403",
            "model endpoint http 404",
        ]
        return any(marker in msg for marker in non_retryable_markers)

    def run_cycle(self, dry_run: bool = False) -> Dict[str, Any]:
        if not self._run_lock.acquire(blocking=False):
            return {"success": False, "status": "skipped", "message": "Run already in progress"}

        started_at = datetime.now(timezone.utc)
        run_timestamp_iso = started_at.isoformat()
        run_id = None
        total_locations = 0
        predictions_made = 0
        alerts_created = 0
        errors_count = 0
        per_user_stats: Dict[str, Dict[str, int]] = {}

        try:
            if self._has_recent_running_run():
                return {"success": False, "status": "skipped", "message": "Recent running job exists"}

            run_id = self._create_prediction_run_row()

            locations = self._fetch_saved_locations()
            total_locations = len(locations)
            self._update_prediction_run_row(run_id, total_locations=total_locations)

            for location in locations:
                user_id = str(location.get("user_id") or "").strip()
                if not user_id:
                    continue
                stats = per_user_stats.setdefault(
                    user_id,
                    {
                        "total_locations": 0,
                        "predictions_made": 0,
                        "alerts_created": 0,
                        "errors_count": 0,
                    },
                )
                stats["total_locations"] += 1

            for batch_start in range(0, total_locations, self.batch_size):
                batch = locations[batch_start : batch_start + self.batch_size]
                for location in batch:
                    user_id = str(location.get("user_id") or "").strip()
                    if user_id:
                        stats = per_user_stats.setdefault(
                            user_id,
                            {
                                "total_locations": 0,
                                "predictions_made": 0,
                                "alerts_created": 0,
                                "errors_count": 0,
                            },
                        )
                        stats["predictions_made"] += 1

                    predictions_made += 1
                    outcome = self.predict_for_location_with_retry(location)
                    if not outcome.success:
                        errors_count += 1
                        if user_id:
                            per_user_stats[user_id]["errors_count"] += 1
                        logger.error(
                            "Prediction failed user=%s location_id=%s city=%s error=%s",
                            user_id or "unknown",
                            str(location.get("id") or "unknown"),
                            str(location.get("city") or "unknown"),
                            outcome.error,
                        )
                        continue

                    if outcome.risk_score < self.risk_threshold:
                        continue

                    location_id = location.get("id")
                    disaster_type = self._normalize_disaster_type(outcome.disaster_type)
                    if not user_id or not location_id:
                        errors_count += 1
                        if user_id:
                            per_user_stats[user_id]["errors_count"] += 1
                        continue

                    if self._alert_exists_recently(user_id, location_id, disaster_type):
                        continue

                    if not dry_run:
                        self._insert_disaster_alert(
                            user_id=user_id,
                            location_id=location_id,
                            disaster_type=disaster_type,
                            risk_score=outcome.risk_score,
                            confidence=outcome.confidence,
                            prediction_timestamp=outcome.prediction_timestamp,
                        )
                    alerts_created += 1
                    per_user_stats[user_id]["alerts_created"] += 1

            duration = (datetime.now(timezone.utc) - started_at).total_seconds()
            self._update_prediction_run_row(
                run_id,
                predictions_made=predictions_made,
                alerts_created=alerts_created,
                errors_count=errors_count,
                duration_seconds=duration,
                status="completed",
            )
            self._insert_user_prediction_runs(
                run_timestamp=run_timestamp_iso,
                per_user_stats=per_user_stats,
                duration_seconds=duration,
                status="completed",
            )

            return {
                "success": True,
                "status": "completed",
                "summary": {
                    "run_id": run_id,
                    "total_locations": total_locations,
                    "predictions_made": predictions_made,
                    "alerts_created": alerts_created,
                    "errors_count": errors_count,
                    "duration_seconds": round(duration, 2),
                    "dry_run": dry_run,
                },
            }

        except Exception as exc:
            logger.exception("Scheduler cycle failed: %s", exc)
            duration = (datetime.now(timezone.utc) - started_at).total_seconds()
            if run_id:
                self._update_prediction_run_row(
                    run_id,
                    total_locations=total_locations,
                    predictions_made=predictions_made,
                    alerts_created=alerts_created,
                    errors_count=max(errors_count, 1),
                    duration_seconds=duration,
                    status="failed",
                )
            self._insert_user_prediction_runs(
                run_timestamp=run_timestamp_iso,
                per_user_stats=per_user_stats,
                duration_seconds=duration,
                status="failed",
            )
            return {
                "success": False,
                "status": "failed",
                "message": str(exc),
            }
        finally:
            self._run_lock.release()

    def predict_for_location_with_retry(self, location: Dict[str, Any]) -> PredictionOutcome:
        last_error = "Unknown prediction error"
        for idx, backoff in enumerate(self.retry_backoff_seconds):
            outcome = self.predict_for_location(location)
            if outcome.success:
                return outcome
            last_error = outcome.error or last_error
            if self._is_non_retryable_error(last_error):
                break
            if idx < len(self.retry_backoff_seconds) - 1:
                time.sleep(backoff)
        self.last_prediction_error = last_error
        self.last_prediction_error_at = datetime.now(timezone.utc).isoformat()
        return PredictionOutcome(success=False, error=last_error)

    def predict_for_location(self, location: Dict[str, Any]) -> PredictionOutcome:
        if self.hazardguard_service is not None:
            return self._predict_with_local_hazardguard(location)

        if not self.model_predict_url:
            return PredictionOutcome(success=False, error="MOBILE_HF_PREDICTION_URL is empty")

        headers = {"Content-Type": "application/json"}
        if self.model_api_token:
            headers["Authorization"] = f"Bearer {self.model_api_token}"

        payload = {
            "latitude": location.get("lat"),
            "longitude": location.get("lon"),
            "city": location.get("city"),
            "country": location.get("country"),
            "forecast_horizon_hours": self.forecast_hours,
        }

        try:
            resp = requests.post(
                self.model_predict_url,
                json=payload,
                headers=headers,
                timeout=self.request_timeout_seconds,
            )
        except Exception as exc:
            return PredictionOutcome(success=False, error=f"Request failed: {exc}")

        if resp.status_code >= 400:
            return PredictionOutcome(success=False, error=f"Model endpoint HTTP {resp.status_code}")

        try:
            result = resp.json()
        except Exception as exc:
            return PredictionOutcome(success=False, error=f"Invalid JSON from model endpoint: {exc}")

        return self._extract_prediction_fields(result)

    def _predict_with_local_hazardguard(self, location: Dict[str, Any]) -> PredictionOutcome:
        lat = location.get("lat")
        lon = location.get("lon")
        if lat is None or lon is None:
            return PredictionOutcome(success=False, error="Location lat/lon missing")

        try:
            result = self.hazardguard_service.predict_disaster_for_location(
                latitude=float(lat),
                longitude=float(lon),
                reference_date=None,
            )
        except Exception as exc:
            return PredictionOutcome(success=False, error=f"Local HazardGuard error: {exc}")

        if not result.get("success"):
            base_error = str(result.get("error", "HazardGuard prediction failed"))
            collection = result.get("data_collection") or {}
            stage_errors: List[str] = []
            if isinstance(collection, dict):
                for stage in ("weather", "features", "raster"):
                    stage_data = collection.get(stage) or {}
                    if isinstance(stage_data, dict) and stage_data.get("success") is False:
                        stage_msg = stage_data.get("message") or stage_data.get("error") or "failed"
                        stage_errors.append(f"{stage}: {stage_msg}")

            detailed_error = base_error
            if stage_errors:
                detailed_error = f"{base_error} | " + " | ".join(stage_errors)

            return PredictionOutcome(success=False, error=detailed_error, raw=result)

        prediction = result.get("prediction") or {}
        probability = prediction.get("probability") or {}
        risk_score = probability.get("disaster")
        confidence = prediction.get("confidence")

        disaster_type = None
        disaster_types_block = result.get("disaster_types") or {}
        if isinstance(disaster_types_block, dict):
            dt_list = disaster_types_block.get("disaster_types")
            if isinstance(dt_list, list) and dt_list:
                disaster_type = dt_list[0]

        try:
            parsed_risk = float(risk_score) if risk_score is not None else 0.0
        except Exception:
            parsed_risk = 0.0

        try:
            parsed_conf = float(confidence) if confidence is not None else parsed_risk
        except Exception:
            parsed_conf = parsed_risk

        if risk_score is None:
            return PredictionOutcome(success=False, error="HazardGuard response missing disaster probability")

        return PredictionOutcome(
            success=True,
            risk_score=parsed_risk,
            confidence=max(0.0, min(1.0, parsed_conf)),
            disaster_type=self._normalize_disaster_type(disaster_type),
            prediction_timestamp=datetime.now(timezone.utc).isoformat(),
            raw=result,
        )

    def _extract_prediction_fields(self, result: Dict[str, Any]) -> PredictionOutcome:
        candidates = [result]
        data = result.get("data")
        if isinstance(data, dict):
            candidates.append(data)
        prediction = result.get("prediction")
        if isinstance(prediction, dict):
            candidates.append(prediction)
        if isinstance(data, dict) and isinstance(data.get("prediction"), dict):
            candidates.append(data["prediction"])

        risk_score = None
        confidence = None
        disaster_type = None

        for item in candidates:
            if risk_score is None:
                risk_score = item.get("risk_score")
            if risk_score is None:
                risk_score = item.get("disaster_probability")
            if risk_score is None and isinstance(item.get("probability"), dict):
                risk_score = item["probability"].get("disaster")

            if confidence is None:
                confidence = item.get("confidence")

            if disaster_type is None:
                disaster_type = item.get("disaster_type") or item.get("predicted_disaster_type")
            if disaster_type is None and isinstance(item.get("disaster_types"), list) and item.get("disaster_types"):
                disaster_type = item["disaster_types"][0]
            if disaster_type is None and isinstance(item.get("prediction"), str):
                label = item["prediction"].upper()
                if label in _ALLOWED_DISASTER_TYPES:
                    disaster_type = label
                elif label in {"DISASTER", "ALERT", "HIGH_RISK"}:
                    disaster_type = "STORM"

        try:
            parsed_risk = float(risk_score) if risk_score is not None else 0.0
        except Exception:
            parsed_risk = 0.0

        try:
            parsed_confidence = float(confidence) if confidence is not None else parsed_risk
        except Exception:
            parsed_confidence = parsed_risk

        if risk_score is None:
            return PredictionOutcome(success=False, error="Missing risk score in model response")

        return PredictionOutcome(
            success=True,
            risk_score=parsed_risk,
            confidence=max(0.0, min(1.0, parsed_confidence)),
            disaster_type=self._normalize_disaster_type(disaster_type),
            prediction_timestamp=datetime.now(timezone.utc).isoformat(),
            raw=result,
        )

    def _normalize_disaster_type(self, disaster_type: Optional[str]) -> str:
        value = (disaster_type or "STORM").upper().strip()
        if value in _ALLOWED_DISASTER_TYPES:
            return value
        if value == "FLOODING":
            return "FLOOD"
        return "STORM"

    def _fetch_saved_locations(self) -> List[Dict[str, Any]]:
        response = (
            self.supabase.table("saved_locations")
            .select("id,user_id,city,country,lat,lon")
            .execute()
        )
        return response.data or []

    def _alert_exists_recently(self, user_id: str, location_id: str, disaster_type: str) -> bool:
        window_start = datetime.now(timezone.utc) - timedelta(hours=self.forecast_hours)
        response = (
            self.supabase.table("disaster_alerts")
            .select("id")
            .eq("user_id", user_id)
            .eq("location_id", location_id)
            .eq("disaster_type", disaster_type)
            .gte("alert_sent_at", window_start.isoformat())
            .limit(1)
            .execute()
        )
        return bool(response.data)

    def _insert_disaster_alert(
        self,
        user_id: str,
        location_id: str,
        disaster_type: str,
        risk_score: float,
        confidence: float,
        prediction_timestamp: str,
    ) -> None:
        self.supabase.table("disaster_alerts").insert(
            {
                "user_id": user_id,
                "location_id": location_id,
                "disaster_type": disaster_type,
                "risk_score": round(float(risk_score), 3),
                "confidence": round(float(confidence), 3),
                "prediction_timestamp": prediction_timestamp,
            }
        ).execute()

    def _insert_user_prediction_runs(
        self,
        run_timestamp: str,
        per_user_stats: Dict[str, Dict[str, int]],
        duration_seconds: float,
        status: str,
    ) -> None:
        rows: List[Dict[str, Any]] = []
        for user_id, stats in per_user_stats.items():
            rows.append(
                {
                    "user_id": user_id,
                    "run_timestamp": run_timestamp,
                    "total_locations": int(stats.get("total_locations", 0)),
                    "predictions_made": int(stats.get("predictions_made", 0)),
                    "alerts_created": int(stats.get("alerts_created", 0)),
                    "errors_count": int(stats.get("errors_count", 0)),
                    "duration_seconds": round(float(duration_seconds), 2),
                    "status": status,
                }
            )

        if rows:
            self.supabase.table("prediction_runs").insert(rows).execute()

    def _create_prediction_run_row(self) -> str:
        response = (
            self.supabase.table("prediction_runs")
            .insert(
                {
                    "run_timestamp": datetime.now(timezone.utc).isoformat(),
                    "total_locations": 0,
                    "predictions_made": 0,
                    "alerts_created": 0,
                    "errors_count": 0,
                    "status": "running",
                }
            )
            .execute()
        )
        rows = response.data or []
        if not rows or not rows[0].get("id"):
            raise RuntimeError("Unable to create prediction_runs record")
        return rows[0]["id"]

    def _update_prediction_run_row(
        self,
        run_id: str,
        total_locations: Optional[int] = None,
        predictions_made: Optional[int] = None,
        alerts_created: Optional[int] = None,
        errors_count: Optional[int] = None,
        duration_seconds: Optional[float] = None,
        status: Optional[str] = None,
    ) -> None:
        update_data: Dict[str, Any] = {}
        if total_locations is not None:
            update_data["total_locations"] = int(total_locations)
        if predictions_made is not None:
            update_data["predictions_made"] = int(predictions_made)
        if alerts_created is not None:
            update_data["alerts_created"] = int(alerts_created)
        if errors_count is not None:
            update_data["errors_count"] = int(errors_count)
        if duration_seconds is not None:
            update_data["duration_seconds"] = round(float(duration_seconds), 2)
        if status is not None:
            update_data["status"] = status

        if update_data:
            self.supabase.table("prediction_runs").update(update_data).eq("id", run_id).execute()

    def _has_recent_running_run(self) -> bool:
        window_start = datetime.now(timezone.utc) - timedelta(hours=2)
        response = (
            self.supabase.table("prediction_runs")
            .select("id")
            .eq("status", "running")
            .gte("run_timestamp", window_start.isoformat())
            .limit(1)
            .execute()
        )
        return bool(response.data)

    def get_runtime_status(self) -> Dict[str, Any]:
        return {
            "risk_threshold": self.risk_threshold,
            "forecast_hours": self.forecast_hours,
            "batch_size": self.batch_size,
            "request_timeout_seconds": self.request_timeout_seconds,
            "retry_backoff_seconds": self.retry_backoff_seconds,
            "has_local_hazardguard": self.hazardguard_service is not None,
            "model_predict_url_set": bool(self.model_predict_url),
            "last_prediction_error": self.last_prediction_error,
            "last_prediction_error_at": self.last_prediction_error_at,
        }
