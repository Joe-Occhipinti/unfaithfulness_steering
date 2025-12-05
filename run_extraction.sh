#!/bin/bash

# Stop on error
set -e

echo "=== Starting Setup ==="

# 1. Install Dependencies
echo "Installing dependencies..."
# pip install --upgrade pip
pip install transformers accelerate bitsandbytes tqdm hf_transfer --break-system-packages

# 2. Verify Data
if [ ! -f "data/off_policy_responses.jsonl" ]; then
    echo "Error: Data file not found at data/off_policy_responses.jsonl"
    echo "Please upload your data file to the 'data' directory."
    exit 1
fi

# 3. Run Extraction
echo "=== Running Extraction ==="
echo "Model: deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
echo "Input: data/off_policy_responses.jsonl"
echo "Output: results/activations_run2"

python extract_last_token_activations.py \
    --model_id "deepseek-ai/DeepSeek-R1-Distill-Llama-8B" \
    --input_file "data/off_policy_responses.jsonl" \
    --output_dir "results/activations_run2"

echo "=== Done! ==="
echo "Results are in results/activations_run2"
