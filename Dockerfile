FROM python:3.11-slim

WORKDIR /app

# System libraries required by OpenCV / EasyOCR
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install --no-cache-dir -e ".[ml,llm,api]"

COPY api ./api
COPY models ./models
COPY data ./data

ENV PYTHONUNBUFFERED=1
EXPOSE 8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
