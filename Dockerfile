# ============================================================
# ShrishtiAI Mobile Backend (HF2) — Hugging Face Spaces (Docker)
# Port: 7860 (required by HF Spaces)
# ============================================================

FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libexpat1 \
    libgomp1 \
    curl \
    git \
    git-lfs \
    && git lfs install \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt ./requirements.txt
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

COPY . .

RUN chmod +x entrypoint.sh

ENV FLASK_ENV=production \
    FLASK_DEBUG=False \
    FLASK_HOST=0.0.0.0 \
    FLASK_PORT=7860 \
    PYTHONUNBUFFERED=1 \
    MODEL_ROOT_PATH=/app/models

EXPOSE 7860

CMD ["./entrypoint.sh"]
