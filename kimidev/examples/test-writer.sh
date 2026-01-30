#!/bin/bash

# Activate conda environment if needed
# source /path/to/conda/etc/profile.d/conda.sh
# conda activate swebench

# Set PYTHONPATH to include the project root
export PYTHONPATH="/mnt/huawei/users/swZheng/proDir2026/Kimi-Dev:$PYTHONPATH"

# Configuration
RESULTS_DIR="results/testwriter/Nemotron-Cascade-8B-Thinking"
STEP1_FILE="/mnt/huawei/users/swZheng/proDir/Kimi-Dev/kimidev/examples/results/testwriter/Nemotron-Cascade-8B-Thinking/batch_output_step1.jsonl"
STEP2_FILE="/mnt/huawei/users/swZheng/proDir/Kimi-Dev/kimidev/examples/results/testwriter/Nemotron-Cascade-8B-Thinking/batch_output_step2.jsonl"
DATASET="/mnt/huawei/users/lfu/datasets/SWE-bench_Verified"
REPO_STRUCTURE="/mnt/huawei/users/lfu/codes/Agentless/artifacts/repo_structure/repo_structures/"
NUM_WORKERS=8

# Temporary directory for splits
SPLIT_DIR="${RESULTS_DIR}/splits_proc"
mkdir -p "$SPLIT_DIR"

echo "Splitting input files into smaller chunks (1000 lines)..."
# Split input file (Step 2 output) into chunks of 1000 lines
split -l 1000 -d --additional-suffix=.jsonl "$STEP2_FILE" "${SPLIT_DIR}/step2_chunk_"

echo "Processing chunks..."

for chunk in ${SPLIT_DIR}/step2_chunk_*.jsonl; do
    basename=$(basename "$chunk")
    echo "Processing chunk: $basename"
    
    python batch_rollout_testwriter.py \
        proc_step2 \
        --dataset "$DATASET" \
        --repostructure_dir "$REPO_STRUCTURE" \
        --input_file "$chunk" \
        --step1_file "$STEP1_FILE" \
        --save_dir "$RESULTS_DIR" \
        --num_workers "$NUM_WORKERS"
        
done

echo "Cleaning up split chunks..."
rm -rf "$SPLIT_DIR"

echo "Done!"
