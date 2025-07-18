#!/bin/bash
# Comprehensive test script for Docker setup

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}Testing ML Pipeline Docker Setup${NC}"
echo "=================================="

# Test 1: Build all Docker targets
echo -e "\n${YELLOW}Test 1: Building Docker images${NC}"
echo "Building base image..."
docker build --target base -t mlpipeline:test-base . --quiet

echo "Building dependencies image..."
docker build --target dependencies -t mlpipeline:test-deps . --quiet

echo "Building GPU dependencies image..."
docker build --target gpu-dependencies -t mlpipeline:test-gpu-deps . --quiet

echo "Building development image..."
docker build --target development -t mlpipeline:test-dev . --quiet

echo "Building production image..."
docker build --target production -t mlpipeline:test-prod . --quiet

echo "Building production GPU image..."
docker build --target production-gpu -t mlpipeline:test-prod-gpu . --quiet

echo -e "${GREEN}✓ All Docker images built successfully${NC}"

# Test 2: Validate docker-compose configuration
echo -e "\n${YELLOW}Test 2: Validating docker-compose configuration${NC}"
docker-compose config --quiet
echo -e "${GREEN}✓ docker-compose configuration is valid${NC}"

# Test 3: Test multi-stage build optimization
echo -e "\n${YELLOW}Test 3: Checking image sizes${NC}"
echo "Image sizes:"
docker images mlpipeline:test-* --format "table {{.Repository}}:{{.Tag}}\t{{.Size}}"

# Test 4: Test container startup (without external dependencies)
echo -e "\n${YELLOW}Test 4: Testing container startup${NC}"
echo "Testing production container..."
CONTAINER_ID=$(docker run -d mlpipeline:test-prod sleep 30)
sleep 3
CONTAINER_STATUS=$(docker inspect --format='{{.State.Status}}' $CONTAINER_ID 2>/dev/null || echo "not_found")
if [ "$CONTAINER_STATUS" = "running" ]; then
    echo -e "${GREEN}✓ Production container started successfully${NC}"
    docker stop $CONTAINER_ID > /dev/null
    docker rm $CONTAINER_ID > /dev/null
else
    echo -e "${RED}✗ Production container failed to start (status: $CONTAINER_STATUS)${NC}"
    docker logs $CONTAINER_ID 2>/dev/null || echo "No logs available"
    docker rm $CONTAINER_ID > /dev/null 2>&1 || true
    exit 1
fi

# Test 5: Test GPU container (if NVIDIA runtime available)
echo -e "\n${YELLOW}Test 5: Testing GPU container availability${NC}"
if docker info | grep -q nvidia; then
    echo "NVIDIA Docker runtime detected, testing GPU container..."
    GPU_CONTAINER_ID=$(docker run -d mlpipeline:test-prod-gpu sleep 30)
    sleep 2
    if docker ps | grep -q $GPU_CONTAINER_ID; then
        echo -e "${GREEN}✓ GPU container started successfully${NC}"
        docker stop $GPU_CONTAINER_ID > /dev/null
        docker rm $GPU_CONTAINER_ID > /dev/null
    else
        echo -e "${RED}✗ GPU container failed to start${NC}"
        exit 1
    fi
else
    echo -e "${YELLOW}⚠ NVIDIA Docker runtime not available, skipping GPU test${NC}"
fi

# Test 6: Test development container with Jupyter
echo -e "\n${YELLOW}Test 6: Testing development container${NC}"
# Find an available port
TEST_PORT=8890
while netstat -ln 2>/dev/null | grep -q ":$TEST_PORT "; do
    TEST_PORT=$((TEST_PORT + 1))
done

DEV_CONTAINER_ID=$(docker run -d -p $TEST_PORT:8888 mlpipeline:test-dev)
sleep 5
CONTAINER_STATUS=$(docker inspect --format='{{.State.Status}}' $DEV_CONTAINER_ID 2>/dev/null || echo "not_found")
if [ "$CONTAINER_STATUS" = "running" ]; then
    echo -e "${GREEN}✓ Development container started successfully${NC}"
    
    # Check if Jupyter is accessible (basic check)
    if docker exec $DEV_CONTAINER_ID pgrep -f jupyter > /dev/null 2>&1; then
        echo -e "${GREEN}✓ Jupyter Lab is running in development container${NC}"
    else
        echo -e "${YELLOW}⚠ Jupyter Lab process not detected${NC}"
    fi
    
    docker stop $DEV_CONTAINER_ID > /dev/null
    docker rm $DEV_CONTAINER_ID > /dev/null
else
    echo -e "${RED}✗ Development container failed to start (status: $CONTAINER_STATUS)${NC}"
    docker logs $DEV_CONTAINER_ID 2>/dev/null || echo "No logs available"
    docker rm $DEV_CONTAINER_ID > /dev/null 2>&1 || true
    exit 1
fi

# Test 7: Test package installation in containers
echo -e "\n${YELLOW}Test 7: Testing package installation${NC}"
echo "Testing Python package import in production container..."
if docker run --rm mlpipeline:test-prod python -c "import mlpipeline; print('Package imported successfully')" > /dev/null 2>&1; then
    echo -e "${GREEN}✓ ML Pipeline package imports successfully${NC}"
else
    echo -e "${RED}✗ ML Pipeline package import failed${NC}"
    exit 1
fi

# Test 8: Test CLI availability
echo -e "\n${YELLOW}Test 8: Testing CLI availability${NC}"
if docker run --rm mlpipeline:test-prod python -m mlpipeline.cli --help > /dev/null 2>&1; then
    echo -e "${GREEN}✓ CLI is accessible${NC}"
else
    echo -e "${RED}✗ CLI is not accessible${NC}"
    exit 1
fi

# Test 9: Test security (non-root user)
echo -e "\n${YELLOW}Test 9: Testing security (non-root user)${NC}"
USER_CHECK=$(docker run --rm mlpipeline:test-prod whoami)
if [ "$USER_CHECK" = "mlpipeline" ]; then
    echo -e "${GREEN}✓ Container runs as non-root user (mlpipeline)${NC}"
else
    echo -e "${RED}✗ Container is running as root user${NC}"
    exit 1
fi

# Test 10: Test health check functionality
echo -e "\n${YELLOW}Test 10: Testing health check${NC}"
if docker run --rm mlpipeline:test-prod python -c "import mlpipeline; print('OK')" > /dev/null 2>&1; then
    echo -e "${GREEN}✓ Health check command works${NC}"
else
    echo -e "${RED}✗ Health check command failed${NC}"
    exit 1
fi

# Cleanup test images
echo -e "\n${YELLOW}Cleaning up test images${NC}"
docker rmi mlpipeline:test-base mlpipeline:test-deps mlpipeline:test-gpu-deps mlpipeline:test-dev mlpipeline:test-prod mlpipeline:test-prod-gpu > /dev/null 2>&1 || true

echo -e "\n${GREEN}🎉 All Docker tests passed successfully!${NC}"
echo -e "${BLUE}Docker setup is ready for deployment${NC}"