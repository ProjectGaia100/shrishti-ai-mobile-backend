"""
NASA POWER Weather Service
Handles weather data fetching from NASA POWER API
"""
import requests
import time
import logging
import os
import math
from datetime import datetime, timedelta
from typing import Dict, Optional, Any, Tuple, List
from models.weather_model import WeatherRequest, WeatherDataModel

class NASAPowerService:
    """Service for fetching weather data from NASA POWER API"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.base_url = "https://power.larc.nasa.gov/api/temporal/daily/point"
        self.open_meteo_archive_url = "https://archive-api.open-meteo.com/v1/archive"
        
        # API settings
        self.max_retries = int(os.getenv("NASA_MAX_RETRIES", "3"))
        self.retry_delay = float(os.getenv("NASA_RETRY_DELAY_SECONDS", "4"))
        self.rate_limit_pause = float(os.getenv("NASA_RATE_LIMIT_PAUSE_SECONDS", "10"))
        self.request_delay = float(os.getenv("NASA_REQUEST_DELAY_SECONDS", "0.2"))
        self.timeout = int(os.getenv("WEATHER_REQUEST_TIMEOUT_SECONDS", "25"))
        self.use_open_meteo_fallback = os.getenv("WEATHER_OPEN_METEO_FALLBACK", "true").lower() == "true"

        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": os.getenv(
                    "WEATHER_USER_AGENT",
                    "ShrishtiAI-HazardGuard/1.0 (+https://huggingface.co/spaces)",
                )
            }
        )

        self.last_provider = "none"
        self.last_error = ""
        
        self.initialized = True

    def _set_last_success(self, provider: str) -> None:
        self.last_provider = provider
        self.last_error = ""

    def _set_last_error(self, error: str) -> None:
        self.last_error = error

    def _build_expected_dates(self, start_date: datetime, end_date: datetime) -> List[str]:
        dates: List[str] = []
        cursor = start_date
        while cursor <= end_date:
            dates.append(cursor.strftime("%Y-%m-%d"))
            cursor += timedelta(days=1)
        return dates

    def _safe_float(self, value: Any) -> Optional[float]:
        if value is None:
            return None
        try:
            parsed = float(value)
            if math.isnan(parsed):
                return None
            return parsed
        except Exception:
            return None

    def _specific_humidity_g_kg(
        self,
        temperature_c: Optional[float],
        humidity_pct: Optional[float],
        pressure_hpa: Optional[float],
    ) -> Optional[float]:
        if temperature_c is None or humidity_pct is None:
            return None
        try:
            pressure = pressure_hpa if pressure_hpa is not None else 1013.25
            # Tetens approximation for vapor pressure in hPa.
            es = 6.112 * math.exp((17.67 * temperature_c) / (temperature_c + 243.5))
            e = max(0.0, min(1.0, humidity_pct / 100.0)) * es
            denominator = pressure - (0.378 * e)
            if denominator <= 0:
                return None
            q_kg_kg = 0.622 * e / denominator
            return q_kg_kg * 1000.0
        except Exception:
            return None

    def _fetch_from_nasa(
        self,
        params: Dict[str, Any],
        request: WeatherRequest,
        start_date: datetime,
        end_date: datetime,
    ) -> Tuple[bool, Dict[str, Any]]:
        last_error = "Unknown NASA POWER error"

        for attempt in range(self.max_retries):
            try:
                time.sleep(self.request_delay)
                self.logger.debug(f"NASA POWER API request (attempt {attempt + 1}): {params}")
                response = self.session.get(self.base_url, params=params, timeout=self.timeout)

                if response.status_code == 429:
                    last_error = (
                        f"NASA POWER rate limited request (429) on attempt {attempt + 1}/{self.max_retries}"
                    )
                    self.logger.warning(last_error)
                    if attempt < self.max_retries - 1:
                        time.sleep(self.rate_limit_pause)
                        continue
                    break

                response.raise_for_status()

                json_data = response.json()
                raw_weather_data = json_data.get("properties", {}).get("parameter", {})
                if not raw_weather_data:
                    last_error = "No weather data returned from NASA POWER API"
                    self.logger.warning(last_error)
                    break

                processed_data = WeatherDataModel.process_raw_data(raw_weather_data, request.days_before)
                validation_result = WeatherDataModel.validate_weather_data(processed_data, request.days_before)

                self.logger.info(
                    "Successfully fetched weather data from NASA POWER: %s fields, %s days",
                    len(processed_data),
                    request.days_before,
                )
                self._set_last_success("nasa_power")

                return True, {
                    "weather_data": processed_data,
                    "validation": validation_result,
                    "metadata": {
                        "provider": "nasa_power",
                        "latitude": request.latitude,
                        "longitude": request.longitude,
                        "disaster_date": request.disaster_date,
                        "start_date": start_date.strftime("%Y-%m-%d"),
                        "end_date": end_date.strftime("%Y-%m-%d"),
                        "days_count": request.days_before,
                        "api_response_time": response.elapsed.total_seconds(),
                        "data_quality": validation_result.get("data_quality", {}),
                    },
                    "status": "success",
                }

            except requests.exceptions.Timeout:
                last_error = f"NASA POWER request timeout after {self.timeout}s"
                self.logger.warning(last_error)
            except requests.exceptions.HTTPError as e:
                response = getattr(e, "response", None)
                status_code = response.status_code if response is not None else "unknown"
                body = response.text[:300] if response is not None and response.text else "No response body"
                last_error = f"NASA POWER HTTP error {status_code}: {body}"
                self.logger.warning(last_error)
            except Exception as e:
                last_error = f"NASA POWER unexpected error: {str(e)}"
                self.logger.warning(last_error)

            if attempt < self.max_retries - 1:
                time.sleep(self.retry_delay)

        self._set_last_error(last_error)
        return False, {
            "error": last_error,
            "status": "nasa_failed",
            "attempts": self.max_retries,
        }

    def _fetch_from_open_meteo(
        self,
        request: WeatherRequest,
        start_date: datetime,
        end_date: datetime,
    ) -> Tuple[bool, Dict[str, Any]]:
        params = {
            "latitude": request.latitude,
            "longitude": request.longitude,
            "start_date": start_date.strftime("%Y-%m-%d"),
            "end_date": end_date.strftime("%Y-%m-%d"),
            "timezone": "UTC",
            "daily": ",".join(
                [
                    "temperature_2m_mean",
                    "temperature_2m_max",
                    "temperature_2m_min",
                    "relative_humidity_2m_mean",
                    "precipitation_sum",
                    "surface_pressure_mean",
                    "shortwave_radiation_sum",
                    "wind_speed_10m_mean",
                    "wind_direction_10m_dominant",
                    "dew_point_2m_mean",
                    "cloud_cover_mean",
                    "et0_fao_evapotranspiration",
                    "soil_moisture_0_to_7cm_mean",
                    "soil_moisture_28_to_100cm_mean",
                ]
            ),
        }

        try:
            response = self.session.get(self.open_meteo_archive_url, params=params, timeout=self.timeout)
            response.raise_for_status()
            payload = response.json()
            daily = payload.get("daily") or {}
            daily_dates = daily.get("time") or []
            if not daily_dates:
                raise ValueError("Open-Meteo response did not include daily.time")

            expected_dates = self._build_expected_dates(start_date, end_date)
            idx_by_date = {d: i for i, d in enumerate(daily_dates)}

            weather_data = {field_name: [] for field_name in WeatherDataModel.WEATHER_FIELDS.values()}

            for date_str in expected_dates:
                idx = idx_by_date.get(date_str)

                def daily_value(name: str) -> Optional[float]:
                    if idx is None:
                        return None
                    arr = daily.get(name) or []
                    if idx >= len(arr):
                        return None
                    return self._safe_float(arr[idx])

                temperature_c = daily_value("temperature_2m_mean")
                humidity_perc = daily_value("relative_humidity_2m_mean")
                wind_10m = daily_value("wind_speed_10m_mean")
                precip_mm = daily_value("precipitation_sum")
                surface_pressure_hpa = daily_value("surface_pressure_mean")
                shortwave_mj = daily_value("shortwave_radiation_sum")
                temp_max_c = daily_value("temperature_2m_max")
                temp_min_c = daily_value("temperature_2m_min")
                dew_point_c = daily_value("dew_point_2m_mean")
                cloud_perc = daily_value("cloud_cover_mean")
                wind_direction = daily_value("wind_direction_10m_dominant")
                evap = daily_value("et0_fao_evapotranspiration")
                soil_top = daily_value("soil_moisture_0_to_7cm_mean")
                soil_root = daily_value("soil_moisture_28_to_100cm_mean")

                surface_pressure_kpa = surface_pressure_hpa / 10.0 if surface_pressure_hpa is not None else None
                # Open-Meteo shortwave radiation sum is MJ/m2/day. Convert to kWh/m2/day to match NASA scale.
                solar_radiation = shortwave_mj / 3.6 if shortwave_mj is not None else None
                specific_humidity = self._specific_humidity_g_kg(
                    temperature_c,
                    humidity_perc,
                    surface_pressure_hpa,
                )

                weather_data["temperature_C"].append(temperature_c)
                weather_data["humidity_perc"].append(humidity_perc)
                weather_data["wind_speed_mps"].append(wind_10m)
                weather_data["precipitation_mm"].append(precip_mm)
                weather_data["surface_pressure_hPa"].append(surface_pressure_kpa)
                weather_data["solar_radiation_wm2"].append(solar_radiation)
                weather_data["temperature_max_C"].append(temp_max_c)
                weather_data["temperature_min_C"].append(temp_min_c)
                weather_data["specific_humidity_g_kg"].append(specific_humidity)
                weather_data["dew_point_C"].append(dew_point_c)
                weather_data["wind_speed_10m_mps"].append(wind_10m)
                weather_data["cloud_amount_perc"].append(cloud_perc)
                weather_data["sea_level_pressure_hPa"].append(surface_pressure_kpa)
                weather_data["surface_soil_wetness_perc"].append(soil_top)
                weather_data["wind_direction_10m_degrees"].append(wind_direction)
                weather_data["evapotranspiration_wm2"].append(evap)
                weather_data["root_zone_soil_moisture_perc"].append(soil_root)

            validation_result = WeatherDataModel.validate_weather_data(weather_data, request.days_before)
            self._set_last_success("open_meteo_archive")

            self.logger.warning(
                "Using Open-Meteo fallback weather provider for lat=%s lon=%s date_range=%s..%s",
                request.latitude,
                request.longitude,
                start_date.strftime("%Y-%m-%d"),
                end_date.strftime("%Y-%m-%d"),
            )

            return True, {
                "weather_data": weather_data,
                "validation": validation_result,
                "metadata": {
                    "provider": "open_meteo_archive",
                    "latitude": request.latitude,
                    "longitude": request.longitude,
                    "disaster_date": request.disaster_date,
                    "start_date": start_date.strftime("%Y-%m-%d"),
                    "end_date": end_date.strftime("%Y-%m-%d"),
                    "days_count": request.days_before,
                    "api_response_time": response.elapsed.total_seconds(),
                    "data_quality": validation_result.get("data_quality", {}),
                },
                "status": "success_fallback",
            }
        except Exception as e:
            error_msg = f"Open-Meteo fallback failed: {str(e)}"
            self._set_last_error(error_msg)
            self.logger.error(error_msg)
            return False, {
                "error": error_msg,
                "status": "open_meteo_failed",
            }
        
    def fetch_weather_data(self, request: WeatherRequest) -> Tuple[bool, Dict[str, Any]]:
        """
        Fetch weather data from NASA POWER API
        
        Args:
            request: Weather data request with coordinates and date
            
        Returns:
            Tuple of (success: bool, result: dict)
        """
        try:
            # Validate request
            validation = request.validate()
            if not validation['valid']:
                return False, {
                    'error': 'Invalid request parameters',
                    'details': validation['errors'],
                    'status': 'validation_error'
                }
            
            # Calculate date range
            disaster_date = datetime.strptime(request.disaster_date, '%Y-%m-%d')
            end_date = disaster_date
            start_date = end_date - timedelta(days=request.days_before - 1)
            
            # Prepare API parameters
            params = {
                "latitude": request.latitude,
                "longitude": request.longitude, 
                "start": start_date.strftime("%Y%m%d"),
                "end": end_date.strftime("%Y%m%d"),
                "community": "RE",
                "format": "JSON",
                "parameters": ','.join(WeatherDataModel.WEATHER_FIELDS.keys())
            }
            
            self.logger.info(f"Fetching weather data for lat={request.latitude}, lon={request.longitude}, "
                           f"date_range={start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")

            nasa_success, nasa_result = self._fetch_from_nasa(params, request, start_date, end_date)
            if nasa_success:
                return True, nasa_result

            nasa_error = nasa_result.get('error', 'NASA fetch failed')

            if self.use_open_meteo_fallback:
                self.logger.warning(
                    "NASA weather fetch failed (%s). Trying Open-Meteo fallback.",
                    nasa_error,
                )
                fallback_success, fallback_result = self._fetch_from_open_meteo(request, start_date, end_date)
                if fallback_success:
                    return True, fallback_result

                fallback_error = fallback_result.get('error', 'Open-Meteo fallback failed')
                return False, {
                    'error': f"NASA failed: {nasa_error}; fallback failed: {fallback_error}",
                    'status': 'all_providers_failed'
                }

            return False, {
                'error': nasa_error,
                'status': nasa_result.get('status', 'nasa_failed')
            }
            
        except Exception as e:
            self.logger.error(f"Critical error in fetch_weather_data: {str(e)}")
            return False, {
                'error': f'Critical service error: {str(e)}',
                'status': 'service_error'
            }
    
    def get_weather_for_coordinates(self, latitude: float, longitude: float, 
                                  disaster_date: str, days_before: int = 60) -> Dict[str, Any]:
        """
        Convenience method to get weather data for coordinates
        
        Args:
            latitude: Latitude coordinate
            longitude: Longitude coordinate 
            disaster_date: Disaster date in YYYY-MM-DD format
            days_before: Number of days before disaster to fetch
            
        Returns:
            Weather data result dictionary
        """
        request = WeatherRequest(
            latitude=latitude,
            longitude=longitude,
            disaster_date=disaster_date,
            days_before=days_before
        )
        
        success, result = self.fetch_weather_data(request)
        return {
            'success': success,
            **result
        }
    
    def batch_fetch_weather_data(self, requests_list: list) -> Dict[str, Any]:
        """
        Fetch weather data for multiple locations (with rate limiting)
        
        Args:
            requests_list: List of WeatherRequest objects
            
        Returns:
            Batch processing results
        """
        results = []
        successful = 0
        failed = 0
        
        self.logger.info(f"Starting batch fetch for {len(requests_list)} locations")
        
        for i, request in enumerate(requests_list):
            self.logger.info(f"Processing batch item {i + 1}/{len(requests_list)}")
            
            success, result = self.fetch_weather_data(request)
            
            results.append({
                'index': i,
                'request': {
                    'latitude': request.latitude,
                    'longitude': request.longitude,
                    'disaster_date': request.disaster_date,
                    'days_before': request.days_before
                },
                'success': success,
                'result': result
            })
            
            if success:
                successful += 1
            else:
                failed += 1
            
            # Rate limiting between requests
            if i < len(requests_list) - 1:  # Don't sleep after last request
                time.sleep(self.request_delay)
        
        return {
            'batch_summary': {
                'total_requests': len(requests_list),
                'successful': successful,
                'failed': failed,
                'success_rate': (successful / len(requests_list)) * 100 if requests_list else 0
            },
            'results': results,
            'status': 'completed'
        }
    
    def get_service_status(self) -> Dict[str, Any]:
        """Get service health and configuration"""
        return {
            'service': 'NASA POWER Weather Service',
            'initialized': self.initialized,
            'base_url': self.base_url,
            'fallback_provider': 'open_meteo_archive' if self.use_open_meteo_fallback else 'disabled',
            'last_provider': self.last_provider,
            'last_error': self.last_error,
            'configuration': {
                'max_retries': self.max_retries,
                'retry_delay': self.retry_delay,
                'rate_limit_pause': self.rate_limit_pause,
                'request_delay': self.request_delay,
                'timeout': self.timeout,
                'open_meteo_fallback_enabled': self.use_open_meteo_fallback
            },
            'supported_fields': list(WeatherDataModel.WEATHER_FIELDS.values()),
            'field_count': len(WeatherDataModel.WEATHER_FIELDS)
        }