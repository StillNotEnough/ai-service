FROM python:3.12-slim

WORKDIR /app

# Копируем только requirements.txt (кэширование)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Создаём юзера ДО копирования файлов
RUN addgroup --system fastapi && adduser --system --ingroup fastapi fastapi

# Копируем код (используйте .dockerignore!)
COPY --chown=fastapi:fastapi . .

RUN apt-get update && apt-get install -y --no-install-recommends curl \
  && rm -rf /var/lib/apt/lists/*

# Переключаемся на непривилегированного юзера
USER fastapi:fastapi

EXPOSE 8000


# Healthcheck
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
  CMD curl -fsS http://localhost:8000/health || exit 1

  
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
