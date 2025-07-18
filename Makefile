# Makefile for ML Pipeline Docker operations

.PHONY: help build build-gpu build-dev up up-gpu up-dev down clean logs test test-docker

# Default target
help:
	@echo "ML Pipeline Docker Commands:"
	@echo "  build      - Build production Docker image"
	@echo "  build-gpu  - Build GPU-enabled production image"
	@echo "  build-dev  - Build development image"
	@echo "  up         - Start production services"
	@echo "  up-gpu     - Start GPU-enabled services"
	@echo "  up-dev     - Start development environment"
	@echo "  up-monitor - Start with monitoring stack"
	@echo "  up-distributed - Start with distributed computing stack"
	@echo "  down       - Stop all services"
	@echo "  clean      - Remove all containers and volumes"
	@echo "  logs       - Show logs from all services"
	@echo "  test       - Run tests in container"
	@echo "  test-docker - Test Docker setup and images"
	@echo "  shell      - Open shell in running container"

# Build targets
build:
	./docker/build.sh --target production

build-gpu:
	./docker/build.sh --target production-gpu

build-dev:
	./docker/build.sh --target development

# Service management
up:
	docker-compose up -d mlpipeline postgres mlflow redis

up-gpu:
	docker-compose --profile gpu up -d mlpipeline-gpu postgres mlflow redis

up-dev:
	docker-compose --profile development up -d mlpipeline-dev postgres mlflow redis

up-monitor:
	docker-compose --profile monitoring --profile development up -d

up-distributed:
	docker-compose --profile distributed up -d dask-scheduler dask-worker ray-head ray-worker

up-distributed-full:
	docker-compose -f docker-compose.yml -f docker-compose.distributed.yml up -d

up-distributed-gpu:
	docker-compose -f docker-compose.yml -f docker-compose.distributed.yml --profile gpu up -d

down:
	docker-compose --profile gpu --profile development --profile monitoring --profile distributed down

# Maintenance
clean:
	docker-compose --profile gpu --profile development --profile monitoring --profile distributed down -v
	docker system prune -f

logs:
	docker-compose logs -f

# Development helpers
test:
	docker-compose exec mlpipeline-dev pytest tests/ -v

test-docker:
	./docker/test-docker-setup.sh

test-distributed:
	./docker/test-distributed-setup.sh

shell:
	docker-compose exec mlpipeline-dev bash

# Database operations
db-shell:
	docker-compose exec postgres psql -U mlpipeline -d mlpipeline

# MLflow UI (if not already exposed)
mlflow-ui:
	@echo "MLflow UI available at: http://localhost:5000"

# Jupyter Lab (for development)
jupyter:
	@echo "Jupyter Lab available at: http://localhost:8888"
	@echo "Token: mlpipeline-dev-token"

# Distributed computing dashboards
dask-dashboard:
	@echo "Dask Dashboard available at: http://localhost:8787"

ray-dashboard:
	@echo "Ray Dashboard available at: http://localhost:8265"

# Scale distributed workers
scale-dask-workers:
	docker-compose --profile distributed up -d --scale dask-worker=4

scale-ray-workers:
	docker-compose --profile distributed up -d --scale ray-worker=4