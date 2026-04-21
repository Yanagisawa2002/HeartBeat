FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    HEARTBEAT_DEVICE=cpu \
    HEARTBEAT_CONFIG_PATH=/app/configs/config.yaml \
    HEARTBEAT_CHECKPOINT_DIR=/app/artifacts/checkpoints

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY artifacts ./artifacts
COPY configs ./configs
COPY sample_inputs ./sample_inputs
COPY src ./src
COPY README.md .

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health').read()"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
