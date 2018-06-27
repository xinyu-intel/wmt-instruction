# Reproduce Sockeye [issues](https://github.com/apache/incubator-mxnet/issues/8532)

## Prepare [conda](https://www.anaconda.com/download/) environment
   
1. `wget https://repo.anaconda.com/archive/Anaconda3-5.2.0-Linux-x86_64.sh`

2. `bash Anaconda-latest-Linux-x86_64.sh`

3. `echo ". ~/anaconda3/etc/profile.d/conda.sh" >> ~/.bashrc`
   
   `source ~/.bashrc`
   
   `conda activate`

## Prepare MXNet

You can use pip or build from source:

### Use pip install

`pip install mxnet-cu??mkl`
(Note: ?? can be 75 80 90 91 92)

### Build from source

1. `git clone --recursive https://github.com/apache/incubator-mxnet.git`

2. `make -j $(nproc) USE_MKLDNN=1 USE_BLAS=mkl USE_OPENCV=1 USE_CUDA=1 USE_CUDNN=1 USE_CUDA_PATH=/usr/local/cuda`

## Reproduce

1. `git clone --recursive https://github.com/xinyu-intel/wmt-instrctuion`

2. `bash train_debug_mkl.sh`

You will get te following errors:

```
OMP: Error #15: Initializing libiomp5.so, but found libiomp5.so already initialized.
OMP: Hint: This means that multiple copies of the OpenMP runtime have been linked into the program. That is dangerous, since it can degrade performance or cause incorrect results. The best thing to do is to ensure that only a single OpenMP runtime is linked into the process, e.g. by avoiding static linking of the OpenMP runtime in any library. As an unsafe, unsupported, undocumented workaround you can set the environment variable KMP_DUPLICATE_LIB_OK=TRUE to allow the program to continue to execute, but that may cause crashes or silently produce incorrect results. For more information, please see http://www.intel.com/software/products/support/.
Aborted
```
