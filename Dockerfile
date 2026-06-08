# BlackBox AutoML - AutoGluon Edition
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies for AutoGluon
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Default command: run the pipeline
ENTRYPOINT ["python", "src/pipeline.py"]
CMD []