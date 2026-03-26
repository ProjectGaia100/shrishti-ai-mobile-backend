# Mobile Backend (HF2)

This is an isolated backend for the mobile app only.

It loads and runs HazardGuard models locally (same model family used by HF1 backend) and accepts scheduler requests.

## Endpoints

- `GET /health`
- `POST /api/mobile/predict`
- `POST /api/scheduler/run-mobile-alerts`

## Security headers

- Mobile prediction endpoint supports optional `X-Mobile-Api-Key`.
- Scheduler endpoint requires `X-Scheduler-Secret`.

## Run locally

```bash
pip install -r requirements.txt
python app.py
```

## Deploy to Hugging Face Space

Use this folder as a Docker Space source.

Required secrets/environment variables:

- `SUPABASE_URL`
- `SUPABASE_SERVICE_ROLE_KEY`
- `SCHEDULER_SECRET`
- `HF_TOKEN` (for downloading private model repo)
- `MODEL_REPO_ID` (default `projectgaia/ShrishtiAI-models`)
- `MODEL_ROOT_PATH` (optional)
- `MOBILE_CLIENT_API_KEY` (optional)

Optional tuning:

- `MOBILE_ALERT_RISK_THRESHOLD`
- `MOBILE_ALERT_FORECAST_HOURS`
- `MOBILE_ALERT_BATCH_SIZE`
- `MOBILE_HF_TIMEOUT_SECONDS`
- `MOBILE_HF_BACKOFF_SECONDS`

Notes:

- On startup, this service downloads models if missing and initializes local HazardGuard inference.
- `POST /api/scheduler/run-mobile-alerts` is protected by `X-Scheduler-Secret`.
- This folder is standalone for HF2 deployment (includes vendored HazardGuard runtime modules).

## HF Space settings

- Space SDK: `Docker`
- Exposed port: `7860`
- Secrets: set the required variables above in Space settings.
