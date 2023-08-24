**Setup the Environment:**


**Build NVSHMEM:**

```shell
# install openmpi
wget https://download.open-mpi.org/release/open-mpi/v4.1/openmpi-4.1.4.tar.bz2
tar -jxf openmpi-4.1.4.tar.bz2
cd openmpi-4.1.4
./configure --prefix=$HOME/opt/openmpi --without-memory-manager --without-libnuma --enable-static
make -j 16 all
make install
# installed in $HOME/opt/openmpi, remove the temporary directories:
rm openmpi-4.1.4.tar.bz2
rm -r openmpi-4.1.4
# update env varaible for MPI
export PATH=$HOME/opt/openmpi/bin:$PATH
export LD_LIBRARY_PATH=$HOME/opt/openmpi/lib:$LD_LIBRARY_PATH


# install nccl
# download the file:
https://developer.nvidia.com/downloads/compute/machine-learning/nccl/secure/2.18.3/agnostic/x64/nccl_2.18.3-1+cuda11.0_x86_64.txz
mkdir -p $HOME/opt/nccl
tar -xvf nccl_2.18.3-1+cuda11.0_x86_64.txz -C $HOME/opt/nccl



# Remove the dependency on libibverbs-devel
Add `verbs.h` in the `nvshmem_src_2.8.0-3/src/include`. (copy paste from: https://kernel.googlesource.com/pub/scm/libs/infiniband/libibverbs/+/stable/include/infiniband/verbs.h)
Edit: line 12 of `nvshmem_src_2.8.0-3/src/comm/transports/ibrc/ibrc.cpp`
Edit: line 13 of `nvshmem_src_2.8.0-3/src/comm/transports/common/transport_ib_common.h`
To:
    `#include "verbs.h" // #include "infiniband/verbs.h"`



# download nvshmem 2.8
wget https://developer.download.nvidia.com/compute/redist/nvshmem/2.8.0/source/nvshmem_src_2.8.0-3.txz
tar xvf nvshmem_src_2.8.0-3.txz
cd nvshmem_src_2.8.0-3
mkdir build

# Environment settings
export NVSHMEM_HOME=${PWD}/build
export NVSHMEM_PREFIX=${NVSHMEM_HOME}
export NVSHMEM_USE_GDRCOPY=0
export NVSHMEM_MPI_SUPPORT=1
export MPI_HOME=$HOME/opt/openmpi
export NVSHMEM_USE_NCCL=0
export NCCL_HOME=$HOME/opt/nccl/nccl_2.18.3-1+cuda11.0_x86_64
export CUDA_HOME=/usr/local/cuda
export NVSHMEM_REMOTE_TRANSPORT=none
export LD_LIBRARY_PATH=$NVSHMEM_HOME/lib:$LD_LIBRARY_PATH


# Build NVSHMEM (may require several minutes)
make -j 16

```



**Correctness Check for NVSHMEM:**

```shell
cd /root/zhengwangmeta/pkgs/nvshmem_src_2.8.0-3
export NVSHMEM_HOME=${PWD}/build
export NVSHMEM_PREFIX=${NVSHMEM_HOME}
export NVSHMEM_USE_GDRCOPY=0
export NVSHMEM_MPI_SUPPORT=1
export MPI_HOME=$HOME/opt/openmpi
export NVSHMEM_USE_NCCL=0
export NCCL_HOME=$HOME/opt/nccl/nccl_2.18.3-1+cuda11.0_x86_64
export CUDA_HOME=/usr/local/cuda
export NVSHMEM_REMOTE_TRANSPORT=none
export LD_LIBRARY_PATH=$NVSHMEM_HOME/lib:$LD_LIBRARY_PATH

cd ../../installation_test

# Compile
/usr/local/cuda/bin/nvcc -rdc=true -ccbin g++ -arch=sm_80 -I $NVSHMEM_HOME/include -I $MPI_HOME/include -I $NCCL_HOME/include -L $NVSHMEM_HOME/lib -L $MPI_HOME/lib -L $NCCL_HOME/lib -lnvshmem -lnvidia-ml -lcuda -lcudart -lmpi installation_test.cu -o installation_test
/usr/local/cuda/bin/nvcc -rdc=true -ccbin g++ -arch=sm_80 -I $NVSHMEM_HOME/include -I $MPI_HOME/include -I $NCCL_HOME/include -L $NVSHMEM_HOME/lib -L $MPI_HOME/lib -L $NCCL_HOME/lib -lnvshmem -lnvidia-ml -lcuda -lcudart -lmpi nvshmemHelloworld.cu -o nvshmemHelloworld

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 NVSHMEM_SYMMETRIC_SIZE=32g OMPI_MCA_plm_rsh_agent=sh mpirun --allow-run-as-root -np 8 ./installation_test 8
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 NVSHMEM_SYMMETRIC_SIZE=32g OMPI_MCA_plm_rsh_agent=sh mpirun --allow-run-as-root -np 8 ./nvshmemHelloworld 8


# Run binary on other machine (compiled on dev server, test on rtp):
1. Install MPI on rtp server.
2. Send the `binary_file` and `$NVSHMEM_HOME/lib/nvshmem_bootstrap_mpi.so.2.8.0` to rtp_server
3. Run the binary with the prefix:
NVSHMEM_BOOTSTRAP=plugin NVSHMEM_BOOTSTRAP_PLUGIN=nvshmem_bootstrap_mpi.so.2.8.0
4. Example commands:
NVSHMEM_BOOTSTRAP=plugin NVSHMEM_BOOTSTRAP_PLUGIN=nvshmem_bootstrap_mpi.so.2.8.0 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 NVSHMEM_SYMMETRIC_SIZE=32g OMPI_MCA_plm_rsh_agent=sh mpirun --allow-run-as-root -np 8 ./installation_test 8
```
