# =============================================================================
# AgentLaboratory Dockerfile
# Based on Kaggle's Python Docker image for reproducible ML environments.
# Supports both CPU-only local testing and GPU-accelerated training.
# =============================================================================
# CPU image (default): gcr.io/kaggle-images/python:latest
# GPU image: gcr.io/kaggle-gpu-images/python:latest
# =============================================================================
ARG BASE_IMAGE=gcr.io/kaggle-images/python:latest
FROM ${BASE_IMAGE}

WORKDIR /app

# Copy requirements and install additional dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Set default environment variables
ENV TOKENIZERS_PARALLELISM=false
ENV COMPUTE_MODE=cpu

# Default command runs the main agent laboratory workflow
CMD ["python", "ai_lab_repo.py", "--yaml-location", "experiment_configs/MATH_agentlab.yaml"]
