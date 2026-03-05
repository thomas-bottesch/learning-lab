FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy training code
COPY train.py .
COPY utils/ ./utils/

# Create artifacts directory
RUN mkdir -p /artifacts

# Set environment variables
ENV PYTHONPATH=/app

# Set environment variables for local testing (these will be overridden in Kubernetes)
ENV MLFLOW_TRACKING_URI=http://mlflow-service.mlflow.svc.cluster.local:5000
ENV LAKEFS_ENDPOINT=http://lakefs.lakefs.svc.cluster.local:8000
ENV MINIO_ENDPOINT=http://minio.minio.svc.cluster.local:9000

# Default command (can be overridden)
CMD ["python", "train.py"]