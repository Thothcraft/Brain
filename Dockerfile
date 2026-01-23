# Optimized Dockerfile for fast Railway deployment
# Uses multi-stage build and minimal dependencies

FROM python:3.11-slim as builder

WORKDIR /app

# Install build dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY requirements-deploy.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip wheel --no-cache-dir --wheel-dir=/wheels -r requirements-deploy.txt

# Production stage - minimal image
FROM python:3.11-slim

WORKDIR /app

# Set environment variables
ENV PYTHONPATH="/app"
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install only runtime dependencies (no build tools)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy wheels from builder and install
COPY --from=builder /wheels /wheels
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir /wheels/* && \
    rm -rf /wheels

# Copy only necessary application code (exclude tests, docs, etc.)
COPY server/ ./server/
COPY run.py .
COPY start.sh .
COPY Procfile .

# Create logs directory and ensure start.sh is executable
RUN mkdir -p logs && chmod +x start.sh

# Expose port
EXPOSE 8000

# Health check with shorter intervals
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Start the application using Python script (handles PORT env var properly)
CMD ["python", "run.py"]
