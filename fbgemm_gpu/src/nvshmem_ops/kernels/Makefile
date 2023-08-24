NVCC = /usr/local/cuda/bin/nvcc

# Compiler flags
ARCH = sm_80

CONDA_ENV_DIR = $(HOME)/local/miniconda3/envs/my_fbgemm/lib/python3.8/site-packages
FBGEMM_DIR = $(HOME)/fbsource/fbcode/deeplearning/fbgemm/fbgemm_gpu
# NCCL = -I $(NCCL_HOME)/include -L $(NCCL_HOME)/lib

INCLUDES = -I $(NVSHMEM_HOME)/include -I $(NCCL_HOME)/include -I $(MPI_HOME)/include -I ./include \
		-I $(HOME)/fbsource/third-party/nlohmann-json/single_include \
		-I $(FBGEMM_DIR) \
		-I $(FBGEMM_DIR)/include \
		-I $(CONDA_ENV_DIR)/torch/include \
		-I $(CONDA_ENV_DIR)/torch/include/torch/csrc/api/include
LIB_DIRS = -L $(NVSHMEM_HOME)/lib \
		-L $(NCCL_HOME)/lib \
		-L $(MPI_HOME)/lib \
		-L $(CONDA_ENV_DIR)/torch/lib \
		-L $(HOME)/local/miniconda3/envs/my_fbgemm/lib

LIBS = -ltorch_cpu -ltorch_cuda -lnvshmem -lnvidia-ml -lcuda -lmpi -lcudart -lc10 -lc10_cuda
FLAGS = -std=c++17 -rdc=true --expt-relaxed-constexpr -Xcompiler -w -O3 -arch=$(ARCH) $(INCLUDES) $(LIB_DIRS) $(LIBS)
TORCH_CXX_FLAGS="-D_GLIBCXX_USE_CXX11_ABI=0"

TARGETS = fbgemm_forward_throughput nvshmem_forward_throughput  correctness_check nccl_correctness_check fbgemm_backward_throughput nvshmem_backward_throughput nvshmem_backward_throughput_unsorting nvshmem_backward_throughput_cache

SRC_FILES = $(wildcard ./src/*.cu)
OBJ_FILES = $(patsubst ./src/%.cu,./lib/%.o,$(SRC_FILES))

all: $(TARGETS)

lib/%.o: src/%.cu
	$(NVCC) $(FLAGS) $(TORCH_CXX_FLAGS) -dc $< -o $@


# TBE forward:
fbgemm_forward_throughput: ./lib/fbgemm_forward_throughput.o ./lib/fbgemm_put_tbe.o
	$(NVCC) $(FLAGS) $(TORCH_CXX_FLAGS) $^ -o $@

nvshmem_forward_throughput: ./lib/nvshmem_forward_throughput.o ./lib/nvshmem_put_tbe.o
	$(NVCC) $(FLAGS) $(TORCH_CXX_FLAGS) $^ -o $@

# correctness check:
correctness_check: ./lib/correctness_check.o ./lib/fbgemm_put_tbe_backward.o
	$(NVCC) $(FLAGS) $(TORCH_CXX_FLAGS) $^ -o $@

nccl_correctness_check: ./lib/nccl_correctness_check.o
	$(NVCC) $(FLAGS) $(TORCH_CXX_FLAGS) $^ -o $@

# TBE backward:
fbgemm_backward_throughput: ./lib/fbgemm_backward_throughput.o ./lib/fbgemm_put_tbe_backward.o
	$(NVCC) $(FLAGS) $(TORCH_CXX_FLAGS) $^ -o $@

nvshmem_backward_throughput: ./lib/nvshmem_backward_throughput.o ./lib/fbgemm_put_tbe_backward.o ./lib/nvshmem_put_tbe_backward.o
	$(NVCC) $(FLAGS) $(TORCH_CXX_FLAGS) $^ -o $@

nvshmem_backward_throughput_unsorting: ./lib/nvshmem_backward_throughput_unsorting.o ./lib/fbgemm_put_tbe_backward.o ./lib/nvshmem_put_tbe_backward_unsorting.o
	$(NVCC) $(FLAGS) $(TORCH_CXX_FLAGS) $^ -o $@

nvshmem_backward_throughput_cache: ./lib/nvshmem_backward_throughput_cache.o ./lib/fbgemm_put_tbe_backward.o ./lib/nvshmem_put_tbe_backward.o
	$(NVCC) $(FLAGS) $(TORCH_CXX_FLAGS) $^ -o $@

nvshmem_backward_throughput_alltoall: ./lib/nvshmem_backward_throughput_alltoall.o ./lib/fbgemm_put_tbe_backward.o ./lib/nvshmem_put_tbe_backward.o
	$(NVCC) $(FLAGS) $(TORCH_CXX_FLAGS) $^ -o $@

# Decoupling:
fbgemm_decoupling_throughput: ./lib/fbgemm_decoupling_throughput.o ./lib/fbgemm_put_tbe_backward.o ./lib/fbgemm_decoupling_backward.o
	$(NVCC) $(FLAGS) $(TORCH_CXX_FLAGS) $^ -o $@

nvshmem_decoupling_throughput: ./lib/nvshmem_decoupling_throughput.o ./lib/fbgemm_put_tbe_backward.o ./lib/fbgemm_decoupling_backward.o ./lib/nvshmem_decoupling_backward.o
	$(NVCC) $(FLAGS) $(TORCH_CXX_FLAGS) $^ -o $@

# set_zero_benchmark:
set_zero_benchmark: ./lib/set_zero_benchmark.o
	$(NVCC) $(FLAGS) $(TORCH_CXX_FLAGS) $^ -o $@

clean:
	rm -f $(TARGETS) $(OBJ_FILES)
