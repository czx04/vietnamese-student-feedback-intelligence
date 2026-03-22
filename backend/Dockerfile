FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements-runtime.txt requirements-app.txt requirements-phobert.txt ./
RUN python -m pip install --upgrade pip && \
    python -m pip install -r requirements-runtime.txt -r requirements-app.txt

COPY backend ./backend
COPY ml ./ml
COPY data/sample_feedback.csv ./data/sample_feedback.csv

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=25s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=3)"

CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"]

FROM runtime AS phobert
RUN apt-get update && \
    apt-get install --no-install-recommends -y openjdk-17-jdk-headless && \
    rm -rf /var/lib/apt/lists/* && \
    python -m pip install "torch>=2.2,<3.0" -r requirements-phobert.txt

FROM runtime AS baseline
