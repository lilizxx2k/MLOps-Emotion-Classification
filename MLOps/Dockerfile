FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app/src


WORKDIR /app

RUN apt-get update && apt-get install -y gcc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY configs/ configs/
COPY models/ models/

#CMD ["python", "-m", "src"]

#docker build -t emotion-mlops .
#CMD ["python", "-m", "src.mlops"]
#docker build -t emotion-mlops .
#CMD ["python", "src/mlops/train.py"]
CMD ["python", "-c", "import mlops; print('Docker M10 check passed')"]

