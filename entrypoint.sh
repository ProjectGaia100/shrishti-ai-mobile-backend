#!/bin/bash
set -e

MODEL_ROOT_PATH=${MODEL_ROOT_PATH:-/app/models}
MODEL_SENTINEL="$MODEL_ROOT_PATH/hazardguard/normal_vs_disaster/normal_vs_disaster_xgboost_model.pkl"

if [ ! -f "$MODEL_SENTINEL" ]; then
    echo "[ENTRYPOINT] HazardGuard models not found, downloading..."
    python download_models.py
    echo "[ENTRYPOINT] Model download complete."
else
    echo "[ENTRYPOINT] Models already present, skipping download."
fi

exec gunicorn app:app \
    --bind 0.0.0.0:7860 \
    --workers 1 \
    --threads 2 \
    --timeout 300 \
    --keep-alive 5 \
    --access-logfile - \
    --error-logfile -
