FROM python:3.13-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --no-deps -r requirements.txt

# Copy the application code
COPY . .

# Create necessary directories including dashboard data directory
RUN mkdir -p /app/data/dashboard && \
    chmod -R 777 /app/data

# Set permissions
RUN chmod +x build.sh run.sh run_in_docker.sh

# Expose ports for both the API and Dashboard
EXPOSE 9500 9501

# Run using Gunicorn
CMD ["./run_in_docker.sh"]