#!/bin/bash
# run.sh - Build and run the CUDA Image Processing Pipeline
# Usage: ./run.sh [num_images] [width] [height]

set -e

NUM_IMAGES=${1:-200}
WIDTH=${2:-256}
HEIGHT=${3:-256}

echo "============================================"
echo "  Building CUDA Image Processing Pipeline"
echo "============================================"

mkdir -p sample_output

nvcc -O2 -o imgpipeline src/gpu_image_processing.cu

echo "Build successful."
echo ""

echo "============================================"
echo "  Running: $NUM_IMAGES images (${WIDTH}x${HEIGHT})"
echo "============================================"

./imgpipeline --num_images $NUM_IMAGES --width $WIDTH --height $HEIGHT --threshold 50

echo ""
echo "Done. Output images saved in sample_output/"
echo "Commit sample_output/ to your repository as proof of execution."
