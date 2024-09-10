#!/bin/bash

# Check if the correct number of arguments is provided
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <python_file.py>"
    exit 1
fi

PYTHON_FILE=$1

# Check if Python file exists
if [ ! -f "$PYTHON_FILE" ]; then
    echo "File $PYTHON_FILE does not exist."
    exit 1
fi

DOCKER_TAG="einsum"

# Define local and container directories
LOCAL_DIR="$(pwd)"
CONTAINER_DIR="/app"

docker build -t "$DOCKER_TAG" .

docker run --rm \
    -v "$LOCAL_DIR:$CONTAINER_DIR" \
    -w "$CONTAINER_DIR" \
    "$DOCKER_TAG" \
    python -u "$PYTHON_FILE"

