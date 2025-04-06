NVCC := nvcc
# Default source file and target
SRC ?= sgemm_v0_gmem.cu
TARGET := build/$(basename $(SRC))
CFLAGS := -O3 -arch=sm_80 -lcublas #-lineinfo -keep

all: $(TARGET)

$(TARGET): $(SRC)
	mkdir -p build
	$(NVCC) $(CFLAGS) -o $(TARGET) $(SRC)

run: 
	rm -f $(TARGET)
	$(MAKE) $(TARGET)
	./$(TARGET)

test: 
	rm -f $(TARGET)
	$(MAKE) $(TARGET) CFLAGS="$(CFLAGS) -DENABLE_CPU_GEMM"
	./$(TARGET)

profile: $(TARGET)
	nsys profile --stats=true --force-overwrite true --output $(TARGET) ./$(TARGET)

# /usr/local/NVIDIA-Nsight-Compute/ncu --set full -f -k sgemm_kernel -c 5 -o $(TARGET) ./$(TARGET) 

clean:
	rm -rf build

.PHONY: all run clean profile test