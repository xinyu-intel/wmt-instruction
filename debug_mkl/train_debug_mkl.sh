#!/bin/bash

source ~/.bashrc

conda activate

#export KMP_DUPLICATE_LIB_OK=TRUE

export PYTHONPATH=../subword-nmt:../sockeye


python -m sockeye.train  -s corpus.tc.BPE.de \
                         -t corpus.tc.BPE.en \
                        -vs newstest2016.tc.BPE.de \
                        -vt newstest2016.tc.BPE.en \
                        --encoder rnn \
                        --decoder rnn \
                        --num-embed 256 \
                        --rnn-num-hidden 512 \
                        --rnn-attention-type dot \
                        --max-seq-len 60 \
                        --decode-and-evaluate 500 \
                        --use-cpu \
                        -o wmt_model_debug_mkl
