#!/bin/bash
# Fast build script for development

set -e

echo "🚀 Starting fast build process..."

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker first."
    exit 1
fi

# Build arguments
DOCKERFILE=${1:-Dockerfile.fast}
TAG=${2:-prism-ai-backend:dev}

echo "📦 Building with $DOCKERFILE as $TAG"

# Enable BuildKit for faster builds
export DOCKER_BUILDKIT=1

# Build with caching
docker build \
    --file $DOCKERFILE \
    --tag $TAG \
    --cache-from python:3.12-slim \
    --cache-from $TAG \
    --build-arg BUILDKIT_INLINE_CACHE=1 \
    .

echo "✅ Build completed successfully!"
echo "🏃‍♂️ To run: docker run -p 8000:8000 $TAG"
echo "🐳 Or use: docker-compose -f docker-compose.dev.yml up"