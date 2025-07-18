#!/bin/bash
# Build script for ML Pipeline Docker images

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Default values
BUILD_TARGET="production"
TAG="latest"
PUSH=false
REGISTRY=""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --target)
            BUILD_TARGET="$2"
            shift 2
            ;;
        --tag)
            TAG="$2"
            shift 2
            ;;
        --push)
            PUSH=true
            shift
            ;;
        --registry)
            REGISTRY="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  --target TARGET    Build target (production, production-gpu, development)"
            echo "  --tag TAG         Docker image tag (default: latest)"
            echo "  --push            Push image to registry after build"
            echo "  --registry REG    Registry URL for pushing"
            echo "  --help            Show this help message"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

# Validate build target
case $BUILD_TARGET in
    production|production-gpu|development)
        ;;
    *)
        echo -e "${RED}Invalid build target: $BUILD_TARGET${NC}"
        echo "Valid targets: production, production-gpu, development"
        exit 1
        ;;
esac

# Set image name
if [[ -n "$REGISTRY" ]]; then
    IMAGE_NAME="$REGISTRY/mlpipeline:$TAG-$BUILD_TARGET"
else
    IMAGE_NAME="mlpipeline:$TAG-$BUILD_TARGET"
fi

echo -e "${GREEN}Building ML Pipeline Docker image...${NC}"
echo -e "${YELLOW}Target: $BUILD_TARGET${NC}"
echo -e "${YELLOW}Image: $IMAGE_NAME${NC}"

# Build the image
docker build \
    --target "$BUILD_TARGET" \
    --tag "$IMAGE_NAME" \
    --file Dockerfile \
    .

echo -e "${GREEN}Build completed successfully!${NC}"

# Push if requested
if [[ "$PUSH" == true ]]; then
    if [[ -z "$REGISTRY" ]]; then
        echo -e "${RED}Registry URL required for push operation${NC}"
        exit 1
    fi
    
    echo -e "${GREEN}Pushing image to registry...${NC}"
    docker push "$IMAGE_NAME"
    echo -e "${GREEN}Push completed successfully!${NC}"
fi

echo -e "${GREEN}Image ready: $IMAGE_NAME${NC}"