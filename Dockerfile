FROM python:3.12-slim

WORKDIR /app

# Системные утилиты (curl нужен для HEALTHCHECK)
RUN apt-get update && apt-get install -y --no-install-recommends curl \
  && rm -rf /var/lib/apt/lists/*

# Кэшируем питон-зависимости
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Для установки пакета (editable) нужен pyproject + исходники
COPY pyproject.toml ./
COPY src ./src
RUN pip install --no-cache-dir -e .

# Юзер
RUN addgroup --system fastapi && adduser --system --ingroup fastapi fastapi
USER fastapi:fastapi

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
  CMD curl -fsS http://localhost:8000/health || exit 1

CMD ["uvicorn", "ai_service.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]