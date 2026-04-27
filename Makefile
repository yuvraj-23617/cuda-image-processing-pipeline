NVCC = nvcc
NVCC_FLAGS = -O2
TARGET = imgpipeline
SRC = src/gpu_image_processing.cu
OUTPUT_DIR = sample_output

.PHONY: all run clean help

all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

$(OUTPUT_DIR):
	mkdir -p $(OUTPUT_DIR)

run: $(TARGET) $(OUTPUT_DIR)
	./$(TARGET) --num_images 200 --width 256 --height 256 --threshold 50

run_large: $(TARGET) $(OUTPUT_DIR)
	./$(TARGET) --num_images 20 --width 1024 --height 1024 --threshold 60

run_verbose: $(TARGET) $(OUTPUT_DIR)
	./$(TARGET) --num_images 200 --width 256 --height 256 --verbose

clean:
	rm -f $(TARGET)
	rm -f $(OUTPUT_DIR)/*.pgm $(OUTPUT_DIR)/*.ppm

help:
	@echo "Targets:"
	@echo "  all         - Build the CUDA image pipeline"
	@echo "  run         - Process 200 images (256x256)"
	@echo "  run_large   - Process 20 large images (1024x1024)"
	@echo "  run_verbose - Process 200 images with per-image output"
	@echo "  clean       - Remove binary and output files"
	@echo "  help        - Show this message"
