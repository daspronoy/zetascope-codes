# Makefile for TBD Interview Problem
# Adjust SM architecture to match your GPU:
#   sm_70 = V100, sm_75 = T4, sm_80 = A100, sm_86 = RTX 3090, sm_89 = RTX 4090, sm_87 = Jetson Orin

NVCC      = nvcc
NVCCFLAGS = -O3 -arch=sm_87 --use_fast_math -lineinfo --allow-unsupported-compiler
TARGET    = tbd_orin
# TARGET    = tbd_solution

.PHONY: all clean data run

all: $(TARGET)

$(TARGET): tbd_orin.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $<

# Generate test data using Python
data:
	python3 tbd_problem.py

# Run full pipeline: generate data then run CUDA solution
run: data $(TARGET)
	./$(TARGET) tbd_frames.bin tbd_params.bin

clean:
	rm -f $(TARGET) tbd_frames.bin tbd_params.bin tbd_reference.txt
