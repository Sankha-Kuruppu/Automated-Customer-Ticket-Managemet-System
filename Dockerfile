FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# 1. Dependencies (CPU-only torch keeps the image small; no GPU needed).
COPY requirements.txt .
RUN pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu \
 && pip install -r requirements.txt

# 2. Bundle the encoder at BUILD time: download the pinned base model, then apply our
#    fine-tuned layer delta (models/encoder_ft_delta.safetensors) to build models/encoder_ft.
#    The running container never needs internet access.
COPY training/download_encoder.py training/apply_delta.py training/
COPY models/encoder_ft_delta.safetensors models/
RUN python training/download_encoder.py \
 && python training/apply_delta.py \
 && rm -rf models/encoder models/encoder_ft/.cache models/encoder_ft/onnx models/encoder_ft_delta.safetensors

# 3. Application code and the trained classifier artifact.
COPY app app
COPY schemas schemas
COPY models/ticket_model.joblib models/metrics.json models/

# Run as non-root (uid 1000 also matches Hugging Face Spaces).
RUN useradd -m -u 1000 appuser && mkdir -p /tmp/tensorforge && chown -R appuser /app /tmp/tensorforge
USER appuser

ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    TOKENIZERS_PARALLELISM=false \
    JOBS_DB_PATH=/tmp/tensorforge/jobs.db

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health').status==200 else 1)"

# Single worker: job state is in SQLite and the worker thread lives in this process.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--no-server-header"]
