FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project code
COPY src/mlops /app/mlops

# Copy models folder
COPY models/ /app/models

# Expose Cloud Run port
EXPOSE 8080

# Use $PORT environment variable dynamically
CMD ["sh", "-c", "uvicorn mlops.api:app --host 0.0.0.0 --port ${PORT:-8080}"]

