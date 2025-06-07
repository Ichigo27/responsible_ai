#!/bin/bash

# Run the Responsible AI service in Docker
# This script runs the service from the Docker image

set -eo pipefail

# Load configuration
APP_VERSION=$(grep '^VERSION:' config/templates/app_config.yaml | cut -d ':' -f 2 | xargs echo -n)
CONTAINER_NAME="responsible_ai"

echo ""
echo "Running $CONTAINER_NAME:$APP_VERSION build"
echo ""

# get old docker image id and container , stop and delete them
docker_image=$(docker ps -a | grep -w "$CONTAINER_NAME" | awk '{print $2}') && [ -n "$docker_image" ] && (docker stop "$CONTAINER_NAME" || true) && docker rm "$CONTAINER_NAME" && docker rmi "$docker_image"

IMAGE_NAME="responsible_ai"
docker load -i "${IMAGE_NAME}_${APP_VERSION}.tar.gz"

PORT=$(grep '^PORT:' config/templates/app_config.yaml | cut -d ':' -f 2 | xargs echo -n)
DASHBOARD_PORT=$(grep -A10 'DASHBOARD:' config/templates/app_config.yaml | grep 'PORT:' | head -n1 | cut -d ':' -f 2 | tr -d ' \t')

# Run the container
echo "Running ${IMAGE_NAME}:${APP_VERSION} on port ${PORT} with dashboard on port ${DASHBOARD_PORT}..."
docker run -idt -p "${PORT}:${PORT}" \
    -p "${DASHBOARD_PORT}:${DASHBOARD_PORT}" \
    -v "$(pwd)/config:/app/config" \
    -v "$(pwd)/logs:/app/logs" \
    -v "$(pwd)/data:/app/data" \
    --name "${CONTAINER_NAME}" \
    "${IMAGE_NAME}:${APP_VERSION}"

echo ""
echo "Started $CONTAINER_NAME:$APP_VERSION build"
echo ""
