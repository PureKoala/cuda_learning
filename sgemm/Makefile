NVCC := nvcc
# Default source file and target
SRC ?= sgemm_v0_global_memory.cu
TARGET := build/$(basename $(SRC))
CFLAGS := -O3 -arch=sm_80 -lineinfo -keep

all: $(TARGET)

$(TARGET): $(SRC)
	mkdir -p build
	$(NVCC) $(CFLAGS) -o $(TARGET) $(SRC)

run: $(TARGET)
	./$(TARGET)

profile: $(TARGET)
	nsys profile --stats=true --force-overwrite true --output $(TARGET) ./$(TARGET)
	# /usr/local/NVIDIA-Nsight-Compute/ncu --set full -f -k sgemm_kernel -c 5 -o $(TARGET) ./$(TARGET) 

clean:
	rm -f $(TARGET) $(TARGET).ncu-rep

.PHONY: all run clean profile