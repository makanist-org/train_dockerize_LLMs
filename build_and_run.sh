#!/bin/bash

# Build the Docker image
echo "Building Docker image..."
docker build -t recession-model .

# Run the container with a test question
echo -e "\nRunning container with test question..."
docker run --rm recession-model "What are the signs of an economic recession?"

# Instructions for manual testing
echo -e "\n\nTo test with your own questions, run:"
echo "docker run --rm recession-model \"Your question here\""