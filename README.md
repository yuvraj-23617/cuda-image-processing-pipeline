# CUDA Image Processing Pipeline

A GPU-accelerated image processing pipeline that batch-processes **200 images** through four CUDA kernels, demonstrating significant speedup over CPU-only processing.

## Project Description

This project implements a 4-stage image processing pipeline using custom CUDA kernels. Each image passes through:

1. **RGB to Grayscale** — Luminance conversion using `Y = 0.299R + 0.587G + 0.114B`
2. **Gaussian Blur (5×5)** — Noise reduction via convolution with a Gaussian kernel
3. **Sobel Edge Detection** — Gradient magnitude computation using Sobel operators
4. **Binary Thresholding** — Converts edge map to clean black/white output

The pipeline processes **200 synthetic test images** (256×256 RGB), each containing randomized geometric shapes (circles, rectangles) on gradient backgrounds. This provides varied input data for meaningful edge detection results.

### GPU Parallelism

Each CUDA kernel maps **one thread per pixel**. For a 256×256 image:
- Grid: 16×16 blocks
- Block: 16×16 threads (256 threads)
- Total: 65,536 threads per kernel
- 4 kernels × 200 images = **800 kernel launches** per run

## Code Structure

```
cuda-image-pipeline/
├── src/
│   └── gpu_image_processing.cu   # All CUDA code (single file)
├── sample_output/                # Output artifacts (proof of execution)
│   ├── input_rgb.ppm
│   ├── stage1_grayscale.pgm
│   ├── stage2_gaussian_blur.pgm
│   ├── stage3_sobel_edges.pgm
│   └── stage4_threshold.pgm
├── Makefile
├── run.sh
└── README.md
```

## Building and Running

### Prerequisites
- NVIDIA GPU with CUDA support
- CUDA Toolkit installed (`nvcc` compiler)

### Quick Start

```bash
# Option 1: Use run.sh
chmod +x run.sh
./run.sh

# Option 2: Use Makefile
make
make run

# Option 3: Manual
nvcc -O2 -o imgpipeline src/gpu_image_processing.cu
./imgpipeline --num_images 200 --width 256 --height 256
```

### CLI Arguments

| Flag | Default | Description |
|------|---------|-------------|
| `--num_images N` | 200 | Number of images to process |
| `--width W` | 256 | Image width in pixels |
| `--height H` | 256 | Image height in pixels |
| `--threshold T` | 50 | Binary threshold (0-255) |
| `--output DIR` | sample_output | Directory for output images |
| `--verbose` | off | Print per-image timing |
| `--help` | — | Show usage |

### Examples

```bash
# Process 200 small images (default)
./imgpipeline --num_images 200 --width 256 --height 256

# Process 20 large images
./imgpipeline --num_images 20 --width 1024 --height 1024

# Custom threshold with verbose output
./imgpipeline --num_images 100 --threshold 80 --verbose
```

### Google Colab

```python
# Cell 1: Write the file
%%writefile gpu_image_processing.cu
# (paste contents of src/gpu_image_processing.cu)

# Cell 2: Compile and run
!mkdir -p sample_output
!nvcc -O2 -o imgpipeline gpu_image_processing.cu
!./imgpipeline --num_images 200

# Cell 3: View outputs
from PIL import Image
import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, 5, figsize=(20, 4))
files = ['sample_output/input_rgb.ppm', 'sample_output/stage1_grayscale.pgm',
         'sample_output/stage2_gaussian_blur.pgm', 'sample_output/stage3_sobel_edges.pgm',
         'sample_output/stage4_threshold.pgm']
titles = ['Input RGB', 'Grayscale', 'Gaussian Blur', 'Sobel Edges', 'Threshold']
for ax, f, t in zip(axes, files, titles):
    ax.imshow(Image.open(f), cmap='gray')
    ax.set_title(t); ax.axis('off')
plt.tight_layout(); plt.show()
```

## Output Artifacts

The program saves pipeline stage images for visual verification in `sample_output/`:

| File | Description |
|------|-------------|
| `input_rgb.ppm` | Original synthetic RGB image |
| `stage1_grayscale.pgm` | After luminance conversion |
| `stage2_gaussian_blur.pgm` | After 5×5 Gaussian smoothing |
| `stage3_sobel_edges.pgm` | After Sobel gradient computation |
| `stage4_threshold.pgm` | Final binary edge map |

## Sample Output

```
==========================================================
  CUDA Image Processing Pipeline
==========================================================
  GPU            : Tesla T4
  SMs            : 40
  Images         : 200  (256 x 256 RGB)
  Threshold      : 50
  Total data     : 37.5 MB
  Pipeline       : Grayscale -> GaussBlur -> Sobel -> Threshold
==========================================================

[GPU] Processing 200 images through 4-kernel pipeline...
  [GPU] Image   1 / 200 done
  [GPU] Image  50 / 200 done
  [GPU] Image 100 / 200 done
  [GPU] Image 150 / 200 done
  [GPU] Image 200 / 200 done
  [GPU] Complete: 312.45 ms total (1.56 ms/image)

[CPU] Processing 200 images (reference)...
  [CPU] Complete: 8934.21 ms total (44.67 ms/image)

==========================================================
  Results Summary
==========================================================
  Images processed : 200  (256 x 256)
  Kernels per image: 4
  Total kernels    : 800
  GPU total time   :     312.45 ms
  CPU total time   :    8934.21 ms
  Speedup          : 28.6x
==========================================================
```

## Algorithms & Kernels

### Kernel 1: Grayscale
Standard luminance formula mapping each RGB pixel to a single intensity value. Each CUDA thread reads 3 bytes (R,G,B) and writes 1 byte.

### Kernel 2: Gaussian Blur
5×5 convolution with a Gaussian kernel (sigma ≈ 1.0). Each thread reads a 5×5 neighborhood (25 reads) and computes a weighted sum. Border pixels use clamped boundary conditions.

### Kernel 3: Sobel Edge Detection
Applies two 3×3 Sobel operators to compute horizontal (Gx) and vertical (Gy) gradients. The gradient magnitude `sqrt(Gx² + Gy²)` highlights edges. Each thread reads a 3×3 neighborhood (9 reads).

### Kernel 4: Binary Threshold
Simple comparison: pixels above the threshold become white (255), below become black (0). This cleans up the edge map for final output.

## Lessons Learned

- **Memory transfer overhead**: For small images, the `cudaMemcpy` host-to-device transfer can dominate kernel execution time. Batching transfers or using pinned memory would improve throughput.
- **Kernel launch overhead**: With 800 kernel launches, the per-launch overhead (~5-10μs each) adds up. Using CUDA streams or fusing kernels would reduce this.
- **Shared memory opportunity**: The Gaussian blur kernel reads overlapping neighborhoods — using shared memory tiling would reduce redundant global memory reads significantly.
- **Speedup scales with image size**: Larger images (1024×1024) show higher speedups because the GPU's parallelism is better utilized with more pixels.
