# Use Python 3.10 slim image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application
COPY . .

# Copy model files from parent directory
# Note: In production, models should be downloaded or mounted separately
COPY ../anomaly_detector /app/models/anomaly_detector
COPY ../inside_air_predictor /app/models/inside_air_predictor
COPY ../safeBite_score /app/models/safeBite_score

# Expose port
EXPOSE 8000

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV HOST=0.0.0.0
ENV PORT=8000

# Run the application
CMD ["python", "run.py"]
