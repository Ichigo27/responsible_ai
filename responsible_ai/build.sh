#!/bin/bash

# Build script for Responsible AI service
# This script builds the Docker image

set -e pipefail

echo "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@"
echo ""
echo "Building docker based build"
echo ""
echo "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@"

# Load configuration
APP_VERSION=$(cat config/templates/app_config.yaml | grep '^VERSION:' | cut -d ':' -f 2 | xargs echo -n)
IMAGE_NAME="responsible_ai"

SAVE_IMAGE=true

for arg in "$@"; do
  if [ "$arg" == "--no-package" ]; then
    SAVE_IMAGE=false
    break
  fi
done

echo "Building ${IMAGE_NAME}:${APP_VERSION}"

# Build the Docker image
docker build --no-cache -t "${IMAGE_NAME}:${APP_VERSION}" -f Dockerfile .


echo ""
echo "Build version : $APP_VERSION is created now saving it."
echo ""

if [ "$SAVE_IMAGE" = true ]; then
    echo "Saving docker image"
    # To save docker container
    docker save "${IMAGE_NAME}:$APP_VERSION" | gzip > "${IMAGE_NAME}_$APP_VERSION.tar.gz"

    mkdir -p "${IMAGE_NAME}/logs"
    mkdir -p "${IMAGE_NAME}/models"
    mv "${IMAGE_NAME}_$APP_VERSION.tar.gz" "${IMAGE_NAME}/"
    cp -r config "${IMAGE_NAME}/"
    cp run.sh "${IMAGE_NAME}/"

    echo "Creating the final build archive"

    # Compile tar.gz file with all the required files
    tar -czf "${IMAGE_NAME}_$APP_VERSION.tar.gz" "${IMAGE_NAME}"

    # Remove config.yaml from the current directory once it has been compiled in tar
    rm -r "${IMAGE_NAME}"
else
  echo "Skipping package creation..."
fi

echo "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@"
echo ""
echo "Created docker based build"
echo ""
echo "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@"
