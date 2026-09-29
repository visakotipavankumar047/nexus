FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY app app
COPY frontend frontend
COPY scripts scripts
COPY data data
RUN mkdir logs && useradd -m nexus && chown -R nexus /app
USER nexus

# Secrets come from the platform's env vars (or `docker run --env-file .env`), never baked into the image.
# PORT is set by Render/Railway/Cloud Run; defaults to 8000 locally.
EXPOSE 8000
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
