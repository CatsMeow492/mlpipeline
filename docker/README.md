# ML Pipeline Docker Setup

This directory contains Docker configuration files for the ML Pipeline project, supporting both CPU and GPU workloads with development and production environments.

## Quick Start

### Prerequisites

- Docker Engine 20.10+
- Docker Compose 2.0+
- For GPU support: NVIDIA Docker runtime

### Basic Usage

```bash
# Start development environment
make up-dev

# Start production environment
make up

# Start GPU-enabled environment
make up-gpu

# Start with monitoring stack
make up-monitor
```

## Architecture

The Docker setup includes the following services:

### Core Services

- **mlpipeline**: Main application (CPU)
- **mlpipeline-gpu**: GPU-enabled application
- **mlpipeline-dev**: Development environment with Jupyter
- **postgres**: PostgreSQL database for MLflow and metadata
- **mlflow**: MLflow tracking server
- **redis**: Caching and task queue

### Monitoring Services (Optional)

- **prometheus**: Metrics collection
- **grafana**: Visualization dashboard

## Build Targets

The Dockerfile uses multi-stage builds with the following targets:

### 1. Production (CPU)
```bash
docker build --target production -t mlpipeline:prod .
```
- Optimized for production workloads
- Minimal image size
- No development tools

### 2. Production GPU
```bash
docker build --target production-gpu -t mlpipeline:prod-gpu .
```
- CUDA toolkit and GPU libraries
- Optimized for GPU workloads
- PyTorch with CUDA support

### 3. Development
```bash
docker build --target development -t mlpipeline:dev .
```
- Jupyter Lab included
- Development tools and debuggers
- Source code mounted for live editing

## Environment Profiles

### Development Profile
```bash
docker-compose --profile development up -d
```
- Jupyter Lab on port 8888
- Live code reloading
- Development databases

### GPU Profile
```bash
docker-compose --profile gpu up -d
```
- GPU-enabled containers
- NVIDIA runtime required
- CUDA device access

### Monitoring Profile
```bash
docker-compose --profile monitoring up -d
```
- Prometheus metrics collection
- Grafana dashboards
- System monitoring

## Configuration

### Environment Variables

Key environment variables for configuration:

```bash
# Database
POSTGRES_HOST=postgres
POSTGRES_DB=mlpipeline
POSTGRES_USER=mlpipeline
POSTGRES_PASSWORD=mlpipeline123

# MLflow
MLFLOW_TRACKING_URI=http://mlflow:5000

# GPU (for GPU containers)
CUDA_VISIBLE_DEVICES=0

# Development
DEBUG=true
LOG_LEVEL=DEBUG
```

### Volume Mounts

Persistent data is stored in Docker volumes:

- `postgres-data`: Database files
- `mlflow-artifacts`: MLflow artifacts
- `jupyter-data`: Jupyter configuration
- `redis-data`: Redis persistence

Local directories mounted for development:

- `./data`: Training and test data
- `./models`: Saved models
- `./configs`: Configuration files
- `./logs`: Application logs

## Usage Examples

### Development Workflow

1. Start development environment:
```bash
make up-dev
```

2. Access Jupyter Lab:
```bash
# Open http://localhost:8888
# Token: mlpipeline-dev-token
```

3. Run experiments in container:
```bash
make shell
python -m mlpipeline.cli train --config configs/example.yaml
```

### Production Deployment

1. Build production image:
```bash
make build
```

2. Start production services:
```bash
make up
```

3. Run pipeline:
```bash
docker-compose exec mlpipeline python -m mlpipeline.cli train --config /app/configs/production.yaml
```

### GPU Training

1. Ensure NVIDIA Docker runtime is installed
2. Start GPU services:
```bash
make up-gpu
```

3. Run GPU training:
```bash
docker-compose exec mlpipeline-gpu python -m mlpipeline.cli train --config /app/configs/gpu-training.yaml
```

## Monitoring and Observability

### MLflow Tracking

- UI available at: http://localhost:5000
- Automatic experiment tracking
- Model registry and versioning

### Prometheus Metrics

- Metrics endpoint: http://localhost:9090
- Custom application metrics
- System resource monitoring

### Grafana Dashboards

- Dashboard UI: http://localhost:3000
- Default credentials: admin/admin123
- Pre-configured Prometheus datasource

## Troubleshooting

### Common Issues

1. **Port conflicts**: Ensure ports 5000, 5432, 8888 are available
2. **GPU access**: Verify NVIDIA Docker runtime installation
3. **Memory issues**: Increase Docker memory limits for large datasets
4. **Permission errors**: Check file ownership and Docker user permissions

### Debugging

```bash
# View logs
make logs

# Access container shell
make shell

# Check service status
docker-compose ps

# Inspect container
docker-compose exec mlpipeline bash
```

### Performance Tuning

1. **Memory**: Adjust container memory limits in docker-compose.yml
2. **CPU**: Set CPU limits and reservations
3. **GPU**: Configure GPU memory fraction for TensorFlow/PyTorch
4. **Storage**: Use SSD volumes for better I/O performance

## Security Considerations

1. **Secrets**: Use Docker secrets or external secret management
2. **Network**: Configure proper network isolation
3. **User permissions**: Run containers as non-root user
4. **Image scanning**: Regularly scan images for vulnerabilities

## Scaling and Production

### Horizontal Scaling

```bash
# Scale ML pipeline workers
docker-compose up -d --scale mlpipeline=3
```

### Load Balancing

Add nginx or traefik for load balancing multiple instances.

### Orchestration

For production deployment, consider:
- Kubernetes with Helm charts
- Docker Swarm mode
- Cloud container services (ECS, GKE, AKS)

## Maintenance

### Regular Tasks

```bash
# Update images
docker-compose pull

# Clean up unused resources
make clean

# Backup data volumes
docker run --rm -v mlpipeline_postgres-data:/data -v $(pwd):/backup alpine tar czf /backup/postgres-backup.tar.gz -C /data .
```

### Health Checks

All services include health checks for monitoring:
- Application health endpoints
- Database connectivity
- Service dependencies