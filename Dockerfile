FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PORT=8000
WORKDIR /app
COPY pyproject.toml README.md ./
COPY src/ ./src/
COPY config/ ./config/
RUN python -m pip install --upgrade pip && python -m pip install -e .
EXPOSE 8000
CMD ["sh", "-c", "exec uvicorn news_predictor.web.app:app --host 0.0.0.0 --port ${PORT:-8000}"]
