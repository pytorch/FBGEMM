# Environment Variable Set up
```
# NVSHMEM and MPI
export NVSHMEM_HOME=/root/zhengwangmeta/pkgs/nvshmem_src_2.8.0-3/build
export NVSHMEM_PREFIX=${NVSHMEM_HOME}
export NVSHMEM_USE_GDRCOPY=0
export NVSHMEM_MPI_SUPPORT=1
export MPI_HOME=$HOME/opt/openmpi
export NVSHMEM_USE_NCCL=0
export NCCL_HOME=$HOME/opt/nccl/nccl_2.18.3-1+cuda11.0_x86_64
export CUDA_HOME=/usr/local/cuda
export NVSHMEM_REMOTE_TRANSPORT=none
export PATH=$HOME/opt/openmpi/bin:$PATH
export LD_LIBRARY_PATH=$HOME/opt/openmpi/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$NCCL_HOME/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$NVSHMEM_HOME/lib:$LD_LIBRARY_PATH
export PATH=$HOME/opt/openmpi/bin:$PATH

# FBGEMM
export LD_LIBRARY_PATH=/root/local/miniconda3/envs/my_fbgemm/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/root/local/miniconda3/envs/my_fbgemm/lib/python3.8/site-packages/torch/lib:$LD_LIBRARY_PATH
```


# Correcntss Check
OMPI_MCA_plm_rsh_agent=sh mpirun --allow-run-as-root -np 8 ./correctness_check <0/1/2>


# Get GPU trace for c++ binary:
nsys profile --trace=cuda,nvtx,mpi --gpu-metrics-device=all -o fbgemm_tbe --export json -f true -x true mpirun --allow-run-as-root -np 8 ./fbgemm_forward_throughput 0 1 0

nsys profile --trace=cuda,nvtx,mpi --gpu-metrics-device=all -o nvshmem_tbe --export json -f true -x true mpirun --allow-run-as-root -np 8 ./nvshmem_forward_throughput 0 0 0


# NCCL All-to-All Correctness Check
OMPI_MCA_plm_rsh_agent=sh mpirun --allow-run-as-root -np 8 ./nccl_correctness_check


# Backward Test
make fbgemm_backward_throughput -j 16
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 OMPI_MCA_plm_rsh_agent=sh mpirun --allow-run-as-root -np 8 ./fbgemm_backward_throughput 0 0 16384

make nvshmem_backward_throughput -j 16
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 OMPI_MCA_plm_rsh_agent=sh mpirun --allow-run-as-root -np 8 ./nvshmem_backward_throughput 0 0 16384

make nvshmem_backward_throughput_unsorting -j 16
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 OMPI_MCA_plm_rsh_agent=sh mpirun --allow-run-as-root -np 8 ./nvshmem_backward_throughput_unsorting 0 0 16384

make nvshmem_backward_throughput_cache -j 16
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 OMPI_MCA_plm_rsh_agent=sh mpirun --allow-run-as-root -np 8 ./nvshmem_backward_throughput_cache 0 0 16384

make nvshmem_backward_throughput_alltoall -j 16
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 OMPI_MCA_plm_rsh_agent=sh mpirun --allow-run-as-root -np 8 ./nvshmem_backward_throughput_alltoall 0 0 16384

OMPI_MCA_plm_rsh_agent=sh mpirun --allow-run-as-root -np 8 ./correctness_check 1



# torchrec api
./torchrec_oemae_throughput.par --nDev=8 --exp=fwd --n_loop=1 --config_file=XXX.yaml

# Decoupling
make fbgemm_decoupling_throughput -j 16
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 OMPI_MCA_plm_rsh_agent=sh mpirun --allow-run-as-root -np 8 ./fbgemm_decoupling_throughput 0 1 1 16384

make nvshmem_decoupling_throughput -j 16
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 OMPI_MCA_plm_rsh_agent=sh mpirun --allow-run-as-root -np 8 ./nvshmem_decoupling_throughput 0 0 16384

OMPI_MCA_plm_rsh_agent=sh mpirun --allow-run-as-root -np 8 ./correctness_check 1


# Set zero benchmark
indices:1327480, total_unique_indices:779593, ratio:0.587273
indices:989195, total_unique_indices:436545, ratio:0.441313
indices:1272562, total_unique_indices:596497, ratio:0.468737
indices:1205758, total_unique_indices:788837, ratio:0.654225
indices:2016857, total_unique_indices:1006164, ratio:0.498877
indices:1970274, total_unique_indices:1318318, ratio:0.669104
indices:1993654, total_unique_indices:1127556, ratio:0.565573
indices:2079790, total_unique_indices:1167374, ratio:0.561294

./set_zero_benchmark 1200000 256
at::empty -- Total (ms): 90.075, avg_kernel latency:0.001 ms/iter
at::zeros -- Total (ms): 42439.184, avg_kernel latency:0.648 ms/iter
empty + zero -- Total (ms): 42429.285, avg_kernel latency:0.647 ms/iter
empty + cudaMemset -- Total (ms): 43087.219, avg_kernel latency:0.657 ms/iter

./set_zero_benchmark 1200000 1
at::empty -- Total (ms): 93.227, avg_kernel latency:0.001 ms/iter
at::zeros -- Total (ms): 448.558, avg_kernel latency:0.007 ms/iter
empty + zero -- Total (ms): 426.863, avg_kernel latency:0.007 ms/iter
empty + cudaMemset -- Total (ms): 293.526, avg_kernel latency:0.004 ms/iter
