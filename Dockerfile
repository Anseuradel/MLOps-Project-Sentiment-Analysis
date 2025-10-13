FROM python:3.11-slim-bullseye

WORKDIR /app

# Install system dependencies first
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc python3-dev && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY src/ ./src
COPY config.py ./config.py
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# COPY . .

# Expose FastAPI port
EXPOSE 8000

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
