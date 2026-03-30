---
title: Mobile Backend
emoji: 🌍
colorFrom: red
colorTo: yellow
sdk: docker
app_port: 7860
pinned: false
---


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
- `WEATHER_OPEN_METEO_FALLBACK` (default `true`)
- `WEATHER_REQUEST_TIMEOUT_SECONDS` (default `25`)
- `NASA_MAX_RETRIES` (default `3`)
- `NASA_RETRY_DELAY_SECONDS` (default `4`)
- `NASA_RATE_LIMIT_PAUSE_SECONDS` (default `10`)

Notes:

- On startup, this service downloads models if missing and initializes local HazardGuard inference.
- `POST /api/scheduler/run-mobile-alerts` is protected by `X-Scheduler-Secret`.
- This folder is standalone for HF2 deployment (includes vendored HazardGuard runtime modules).

## HF Space settings

- Space SDK: `Docker`
- Exposed port: `7860`
- Secrets: set the required variables above in Space settings.

## GitHub CI/CD (repo -> HF2)

This repo includes a GitHub Actions workflow at `.github/workflows/deploy-hf.yml`.

On every push to `main`, it deploys this repo to your HF2 Space.

Required GitHub repository secrets:

- `HF_TOKEN`: Hugging Face write token
- `HF_SPACE_REPO`: Space slug, for example `shrishtiai/ShrishtiAI-mobile-backend`

You can also trigger deployment manually with `workflow_dispatch`.
