# Multi-stage Dockerfile for ML Pipeline
# Stage 1: Base Python environment
FROM python:3.11-slim AS base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd --create-home --shell /bin/bash mlpipeline
WORKDIR /app
RUN chown mlpipeline:mlpipeline /app

# Stage 2: Dependencies installation
FROM base AS dependencies

# Copy requirements files
COPY requirements.txt requirements-dev.txt ./

# Install Python dependencies
RUN pip install --upgrade pip setuptools wheel && \
    pip install -r requirements.txt

# Stage 3: GPU-enabled variant
FROM dependencies AS gpu-dependencies

# Install GPU monitoring tools (CUDA toolkit will be provided by base image or runtime)
RUN apt-get update && apt-get install -y \
    wget \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-gpu.txt ./
RUN pip install -r requirements-gpu.txt

# Stage 4: Development environment
FROM dependencies AS development

# Install development dependencies
RUN pip install -r requirements-dev.txt

# Install Jupyter extensions (optional, ignore failures)
RUN jupyter lab build || echo "Jupyter lab build failed, continuing..."

# Copy source code
COPY --chown=mlpipeline:mlpipeline . .

# Switch to non-root user
USER mlpipeline

# Expose ports for Jupyter and MLflow
EXPOSE 8888 5000

# Default command for development
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]

# Stage 5: Production environment (CPU)
FROM dependencies AS production

# Copy source code
COPY --chown=mlpipeline:mlpipeline . .

# Install the package
RUN pip install -e .

# Switch to non-root user
USER mlpipeline

# Expose MLflow port
EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD python -c "import mlpipeline; print('OK')" || exit 1

# Default command
CMD ["python", "-m", "mlpipeline.cli", "--help"]

# Stage 6: Production environment (GPU)
FROM gpu-dependencies AS production-gpu

# Copy source code
COPY --chown=mlpipeline:mlpipeline . .

# Install the package
RUN pip install -e .

# Switch to non-root user
USER mlpipeline

# Expose MLflow port
EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD python -c "import mlpipeline; print('OK')" || exit 1

# Default command
CMD ["python", "-m", "mlpipeline.cli", "--help"]