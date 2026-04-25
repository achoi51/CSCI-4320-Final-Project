# Parallelized Computational Eigensolver - Makefile
#
# Build modes:
#   make              -> Hybrid MPI + CUDA build (AiMOS / V100)
#   make cpu          -> MPI-only CPU build (no CUDA required)
#   make clean        -> Remove build artifacts

TARGET    := eigensolver
SRC_DIR   := src
BUILD_DIR := build

# Toolchain (override with `make MPICC=... NVCC=...`)
MPICC  ?= mpicc
MPICXX ?= mpicxx
NVCC   ?= nvcc

# V100 = SM 70
CUDA_ARCH ?= sm_70

CFLAGS    := -O3 -Wall -Wextra -std=c99
CXXFLAGS  := -O3 -Wall -Wextra -std=c++11
NVCCFLAGS := -O3 -arch=$(CUDA_ARCH) --use_fast_math -Xcompiler -fPIC

C_SOURCES   := $(wildcard $(SRC_DIR)/*.c)
CU_SOURCES  := $(wildcard $(SRC_DIR)/*.cu)

C_OBJS_GPU  := $(patsubst $(SRC_DIR)/%.c,$(BUILD_DIR)/%.o,$(C_SOURCES))
CU_OBJS_GPU := $(patsubst $(SRC_DIR)/%.cu,$(BUILD_DIR)/%.cu.o,$(CU_SOURCES))

C_OBJS_CPU  := $(patsubst $(SRC_DIR)/%.c,$(BUILD_DIR)/%.cpu.o,$(C_SOURCES))
CU_OBJS_CPU := $(patsubst $(SRC_DIR)/%.cu,$(BUILD_DIR)/%.cucpu.o,$(CU_SOURCES))

# Default target: hybrid build
all: gpu

gpu: CFLAGS    += -DUSE_CUDA
gpu: CXXFLAGS  += -DUSE_CUDA
gpu: NVCCFLAGS += -DUSE_CUDA
gpu: $(BUILD_DIR)/$(TARGET)

cpu: $(BUILD_DIR)/$(TARGET)_cpu

$(BUILD_DIR)/$(TARGET): $(C_OBJS_GPU) $(CU_OBJS_GPU) | $(BUILD_DIR)
	$(MPICXX) $(CXXFLAGS) $^ -o $@ -L$(CUDA_HOME)/lib64 -lcudart -lm

$(BUILD_DIR)/$(TARGET)_cpu: $(C_OBJS_CPU) $(CU_OBJS_CPU) | $(BUILD_DIR)
	$(MPICXX) $(CXXFLAGS) $^ -o $@ -lm

# GPU pipeline: .c via mpicc, .cu via nvcc
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.c | $(BUILD_DIR)
	$(MPICC) $(CFLAGS) -I$(SRC_DIR) -c $< -o $@

$(BUILD_DIR)/%.cu.o: $(SRC_DIR)/%.cu | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) -I$(SRC_DIR) -ccbin=$(MPICXX) -c $< -o $@

# CPU pipeline: .c via mpicc (separate object), .cu reinterpreted as C++ via mpicxx
$(BUILD_DIR)/%.cpu.o: $(SRC_DIR)/%.c | $(BUILD_DIR)
	$(MPICC) $(CFLAGS) -I$(SRC_DIR) -c $< -o $@

$(BUILD_DIR)/%.cucpp.cpp: $(SRC_DIR)/%.cu | $(BUILD_DIR)
	cp $< $@

$(BUILD_DIR)/%.cucpu.o: $(BUILD_DIR)/%.cucpp.cpp
	$(MPICXX) $(CXXFLAGS) -I$(SRC_DIR) -c $< -o $@

$(BUILD_DIR):
	mkdir -p $@

clean:
	rm -rf $(BUILD_DIR)

.PHONY: all gpu cpu clean
