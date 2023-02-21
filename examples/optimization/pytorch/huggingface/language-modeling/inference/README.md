# LLMs Inference
  - [Introduction] (#introduction)
  - [Models] (#model)
    - [GPT-J] (#gpt-j)
      - [Setup] (#setup)
      - [Performance] (#performance)
      - [Accuracy] (#accuracy)
    - [Bloom-176B] (#bloom-176b)
      - [Setup] (#setup)
      - [Performance] (#performance)
      - [Accuracy] (#accuracy)

## Introduction
This document describes the step-by-step instructions to run large language models(LLMs) on 4th Gen Intel® Xeon® Scalable Processor (codenamed [Sapphire Rapids](https://www.intel.com/content/www/us/en/products/docs/processors/xeon-accelerated/4th-gen-xeon-scalable-processors.html)) with PyTorch and [Intel® Extension for PyTorch](https://github.com/intel/intel-extension-for-pytorch).

We now support two models, and we are adding more models and more advanced techniques(distributed inference, model compressions etc.) to better unleash LLM inference on Intel platforms.

- GPT-J
  script `run_gptj.py` is based on [EleutherAI/gpt-j-6B](https://huggingface.co/EleutherAI/gpt-j-6B) and provides inference benchmarking. For [EleutherAI/gpt-j-6B](https://huggingface.co/EleutherAI/gpt-j-6B) quantization, please refer to [quantization example](../quantization/inc)
- BLOOM-176B
  script `run_bloom.py` is adapted from [HuggingFace/transformers-bloom-inference](https://github.com/huggingface/transformers-bloom-inference/blob/main/bloom-inference-scripts/bloom-accelerate-inference.py). 

## Models
### GPT-J
#### Setup
```
# Create Environment (conda)
conda install mkl mkl-include -y
conda install jemalloc -c conda-forge -y
pip install -r requirements.txt
# install pytorch
pip install torch==1.13.1 --extra-index-url https://download.pytorch.org/whl/cpu
# install intel-extension-for-pytorch
git submodule update --init --recursive
cd intel-extension-for-pytorch
python setup.py install
cd ..

# Setup Environment Variables
export KMP_BLOCKTIME=1
export KMP_SETTINGS=1
export KMP_AFFINITY=granularity=fine,compact,1,0
# IOMP
export OMP_NUM_THREADS=< Cores number to use >
export LD_PRELOAD=${LD_PRELOAD}:${CONDA_PREFIX}/lib/libiomp5.so
# Jemalloc is a recommended malloc implementation that emphasizes fragmentation avoidance and scalable concurrency support.
export LD_PRELOAD=${LD_PRELOAD}:${CONDA_PREFIX}/lib/libjemalloc.so
export MALLOC_CONF="oversize_threshold:1,background_thread:true,metadata_thp:auto,dirty_decay_ms:9000000000,muzzy_decay_ms:9000000000"
```

#### Performance
The default search algorithm is beam search with num_beams = 4, if you'd like to use greedy search for comparison, add "--greedy" in args.
##### Single node
```bash
numactl -m <node N> -C <cpu list> \
    python run_gptj.py \
        --precision <fp32/bf16> \
        --max-new-tokens 32
```
##### Multiple nodes
```
WIP
```
#### Accuracy
```
WIP
```

### BLOOM-176B
#### Setup
```
# Create Environment (conda)
conda install mkl mkl-include -y
conda install jemalloc -c conda-forge -y
pip install -r requirements.txt
# install pytorch
pip install torch==1.13.1 --extra-index-url https://download.pytorch.org/whl/cpu
# install intel-extension-for-pytorch
git submodule update --init --recursive
cd intel-extension-for-pytorch
python setup.py install
cd ..

# Setup Environment Variables
export KMP_BLOCKTIME=1
export KMP_SETTINGS=1
export KMP_AFFINITY=granularity=fine,compact,1,0
# IOMP
export OMP_NUM_THREADS=< Cores number to use >
export LD_PRELOAD=${LD_PRELOAD}:${CONDA_PREFIX}/lib/libiomp5.so
# Don't enable jemalloc here since BLOOM-176B requires lots of memory and will have memory contention w/ jemalloc.
```

#### Performance
##### Single node
```bash
numactl -m <node N> -C <cpu list> python3 run_bloom.py --batch_size 1 --benchmark
```
##### Multiple nodes
```bash
WIP
```
#### Accuracy
```
WIP
```


>**Note**: Inference performance speedup with Intel DL Boost (VNNI/AMX) on Intel(R) Xeon(R) hardware, Please refer to [Performance Tuning Guide](https://intel.github.io/intel-extension-for-pytorch/cpu/latest/tutorials/performance_tuning/tuning_guide.html) for more optimizations.
