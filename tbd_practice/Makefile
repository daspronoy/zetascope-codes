# Makefile for TBD Interview Problem
# Adjust SM architecture to match your GPU:
#   sm_70 = V100 (Volta), sm_75 = T4 (Turing), sm_80 = A100 (Ampere), sm_86 = RTX 3090 (Ampere), sm_89 = RTX 4090 (Ada Lovelace), sm_87 = Jetson Orin (Ampere), sm_120 = 5070Ti (Blackwell)

NVCC      = nvcc
NVCCFLAGS = -O3 -arch=sm_87 --use_fast_math -lineinfo --allow-unsupported-compiler
TARGET    = tbd_orin_optimized_v2
# TARGET    = tbd_solution

.PHONY: all clean data run

all: $(TARGET)

$(TARGET): tbd_orin_optimized_v2.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $<

# Generate test data using Python
data:
	python3 tbd_problem.py

# Run full pipeline: generate data then run CUDA solution
run: data $(TARGET)
	./$(TARGET) tbd_frames.bin tbd_params.bin

clean:
	rm -f $(TARGET) tbd_frames.bin tbd_params.bin tbd_reference.txt
