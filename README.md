[Sockeye](https://github.com/awslabs/sockeye) is a sequence-to-sequence framework for Neural Machine Translation based on [Apache MXNet Incubating](https://github.com/apache/incubator-mxnet).

It implements state-of-the-art encoder-decoder architectures, especially attention model:
- Transformer Models with self-attention [[Vaswani et al, '17](https://arxiv.org/abs/1706.03762)]

This document will give you a brief instruction to use sockeye to build GNMT model based on MXNet-MKL.

## Build MXNet with MKLDNN support from source

### Ubuntu pre-req

```
apt-get update && apt-get install -y build-essential git libopencv-dev curl gcc libopenblas-dev python python-pip python-dev python-opencv graphviz python-scipy python-sklearn
```

### Clone MXNet sources

```
git clone --recursive https://github.com/apache/incubator-mxnet.git
cd incubator-mxnet
git submodule update --recursive --init
```

### Build MXNet with MKLDNN

```
make -j $(nproc) USE_OPENCV=1 USE_MKLDNN=1 USE_BLAS=openblas USE_PROFILER=1
```

### Verify MXNet with python

```
export PYTHONPATH=~/incubator-mxnet/python
pip install --upgrade pip 
pip install --upgrade jupyter graphviz cython pandas bokeh matplotlib opencv-python requests
python -c "import mxnet as mx;print((mx.nd.ones((2, 3))*2).asnumpy());"

Expected Output:

[[ 2.  2.  2.]
 [ 2.  2.  2.]]
```
 
## Build GNMT with Sockeye

### Clone Sockeye sources

```
git clone --recursive https://github.com/awslabs/sockeye
```

### Verify Sockeye

```
export PYTHONPATH=~/incubator-mxnet/python:~/sockeye/
python -c "import mxnet as mx;import sockeye;"
```

### Build Attention Model

Follow the [toturial](https://github.com/awslabs/sockeye/tree/master/tutorials/wmt) to build wmt based on transformer encoder/decoder.

When the preprocessing finished, you can reference to the following script when training attention model:

```
export OMP_NUM_THREADS=$num of cpu cores
python -m sockeye.train  -s corpus.tc.BPE.de \
                         -t corpus.tc.BPE.en \
                         -vs newstest2016.tc.BPE.de \
                         -vt newstest2016.tc.BPE.en \
                         --batch-type=word \
                         --batch-size=4096 \
                         --embed-dropout=0:0 \
                         --encoder=transformer \
                         --decoder=transformer \
                         --num-layers=6:6 \
                         --transformer-model-size=512 \
                         --transformer-attention-heads=8 \
                         --transformer-feed-forward-num-hidden=2048 \
                         --transformer-preprocess=n \
                         --transformer-postprocess=dr \
                         --transformer-dropout-attention=0.1 \
                         --transformer-dropout-prepost=0.1 \
                         --transformer-positional-embedding-type fixed \
                         --label-smoothing 0.1 \
                         --weight-tying \
                         --weight-tying-type=src_trg_softmax \
                         --num-embed 512:512 \
                         --initial-learning-rate=0.0001 \
                         --learning-rate-reduce-num-not-improved=8 \
                         --learning-rate-reduce-factor=0.7 \
                         --weight-init xavier --weight-init-scale 3.0 \
                         --weight-init-xavier-factor-type avg \
                         --optimized-metric bleu \
                         --use-cpu \
                         -o wmt_model_transformer
```

You can use mxboard to monitor training progress.

The training progress may consume a lot of time on cpu and you can just train small epoch for inference.

### Profiling

After training you may get a wmt model in `wmt_model_transformer` and you can use this model to do inference. Before performing inference, you should enable mxnet profiling tools first.

```
export OMP_NUM_THREADS=$num of cpu cores
export MXNET_EXEC_BULK_EXEC_INFERENCE=0
export MXNET_EXEC_BULK_EXEC_TRAIN=0
export MXNET_PROFILER_AUTOSTART=1 
```

**Inference:**

```
python -m sockeye.translate -m wmt_model_transformer -i newstest2016.tc.BPE.de -o my_2016.tc.BPE.en --batch-size 32 --output-type benchmark --use-cpu
```

You can try different batch size and different datasets when doing inference.

When inference finished, it will cost some time to save the benchmark file `profile.json`.

**Analyze Profile:**

You can use [mxProfileParser](https://github.com/TaoLv/mxProfileParser) to analyze profile and get the summary.
