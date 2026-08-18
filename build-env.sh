# vLLM-Pascal Build-Umgebung (Ubuntu 24.04, CUDA 12.1, Torch 2.5.1)
# Voraussetzungen: sudo apt install python3.12-dev gcc-12 g++-12
export CUDA_HOME=/usr/local/cuda-12.1
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export TORCH_CUDA_ARCH_LIST="6.0"
export MAX_JOBS=$(nproc)
export CC=/usr/bin/gcc-12
export CXX=/usr/bin/g++-12
export CUDAHOSTCXX=/usr/bin/g++-12
