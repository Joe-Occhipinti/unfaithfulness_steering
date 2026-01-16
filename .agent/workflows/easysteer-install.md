---
description: How to install EasySteer on RunPod (verified method)
---

# Full RunPod Setup for unfaithfulness_steering + EasySteer

> **CRITICAL:** The correct repo is `ZJU-REAL/EasySteer`, NOT `THUDM/EasySteer`.

## Complete Setup (Copy-Paste Ready)

```bash
# 1. Create and activate conda environment (REQUIRED - Python 3.10)
conda create -n easysteer python=3.10 -y
conda activate easysteer

# 2. Clone EasySteer with submodules
cd /workspace
git clone --recurse-submodules https://github.com/ZJU-REAL/EasySteer.git

# 3. Install vllm-steer with pre-compiled wheel (fast, no build)
cd EasySteer/vllm-steer
export VLLM_PRECOMPILED_WHEEL_COMMIT=72506c98349d6bcd32b4e33eec7b5513453c1502
VLLM_USE_PRECOMPILED=1 pip install --editable .

# 4. Install EasySteer
cd ..
pip install --editable .

# 5. Verify vllm is installed
pip show vllm

# 6. Clone your project
cd /workspace
git clone https://github.com/Joe-Occhipinti/unfaithfulness_steering.git

# 7. Install project dependencies
pip install datasets huggingface-hub numpy pandas tqdm matplotlib seaborn scikit-learn scipy gguf
```

## Run Random Mode Steering

```bash
cd /workspace/unfaithfulness_steering

# DeepSeek-R1-Distill-Llama-8B
python eval_steering_easysteer.py \
    --mode random \
    --model "deepseek-ai/DeepSeek-R1-Distill-Llama-8B" \
    --input-file "data/definitive_pipeline_data/DeepSeek-R1-Distill-Llama-8B/faithfulness_annotated_DeepSeek-R1-Distill-Llama-8B_2026-01-10.jsonl" \
    --layers 15 \
    --coefficients 0.6 -0.6 1 -1 2 -2

# Qwen3-14B
python eval_steering_easysteer.py \
    --mode random \
    --model "Qwen/Qwen3-14B" \
    --input-file "data/definitive_pipeline_data/Qwen3-14B/faithfulness_annotated_Qwen3-14B_2026-01-08.jsonl" \
    --layers 15 \
    --coefficients 0.6 -0.6 1 -1 2 -2

# Qwen3-32B
python eval_steering_easysteer.py \
    --mode random \
    --model "Qwen/Qwen3-32B" \
    --input-file "data/definitive_pipeline_data/Qwen3-32B/faithfulness_annotated_Qwen3-32B_2025-12-29.jsonl" \
    --layers 15 \
    --coefficients 0.6 -0.6 1 -1 2 -2
```

## If Pre-compiled Fails: Build from Source

```bash
conda create -n easysteer python=3.10 -y
conda activate easysteer

cd /workspace
git clone --recurse-submodules https://github.com/ZJU-REAL/EasySteer.git
cd EasySteer/vllm-steer
python use_existing_torch.py

# Set GPU architecture: "8.0" for A100, "9.0" for H100/H200
export TORCH_CUDA_ARCH_LIST="9.0"
export CMAKE_ARGS="-DTORCH_CUDA_ARCH_LIST=9.0"
export VLLM_TARGET_DEVICE="cuda"
export MAX_JOBS=8
export CMAKE_BUILD_PARALLEL_LEVEL=8

pip install -r requirements/build.txt
pip install -e . --no-build-isolation -v

cd ..
pip install -e .
```

---
*EasySteer version: compatible with vLLM v0.13.0*
*Created after wasting €10 on wrong installation commands. Never again.*
