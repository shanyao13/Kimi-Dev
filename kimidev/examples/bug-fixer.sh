#!/bin/bash

# Activate conda environment if needed
# source /path/to/conda/etc/profile.d/conda.sh
# conda activate swebench

# Set PYTHONPATH to include the project root
export PYTHONPATH="/mnt/huawei/users/swZheng/proDir/Kimi-Dev:$PYTHONPATH"
export MODEL_NAME="q3-8b"
# export INPUT_FILE="results/bugfixer/${MODEL_NAME}/test/batch_output_step1.jsonl"
# export OUTPUT_FILE="results/bugfixer/${MODEL_NAME}/test/batch_output_step1.jsonl"

# # #### Stage 1: Generate Relevant File Requests
# python3 batch_rollout_bugfixer.py \
#     gen_step1 \
#     --dataset /mnt/huawei/users/lfu/datasets/SWE-bench_Verified \
#     --repostructure_dir /mnt/huawei/users/lfu/codes/Agentless/artifacts/repo_structure/repo_structures/ \
#     --output_file results/bugfixer/q3-8b/batch_requests_step1.jsonl \
#     --model_name "q3-8b" \
#     --temp 0.6 \
#     --passk 40 \
#     --max_tokens 45000\
#     --thinking_budget 45000


#### Stage 2: Generate Repair Requests
python batch_rollout_bugfixer.py \
    proc_step1_gen_step2 \
    --dataset /mnt/huawei/users/lfu/datasets/SWE-bench_Verified \
    --repostructure_dir /mnt/huawei/users/lfu/codes/Agentless/artifacts/repo_structure/repo_structures/ \
    --input_file results/bugfixer/${MODEL_NAME}/batch_output_step1.jsonl \
    --output_file results/bugfixer/${MODEL_NAME}/batch_requests_step2.jsonl \
    --model_name ${MODEL_NAME} \
    --temp 0.6 \
    --num_workers 40 \
    --max_tokens 45000\
    --thinking_budget 45000 \
    --enable_thinking

# #### Stage 3: Process Repairs and Save
# python batch_rollout_bugfixer.py \
#     proc_step2 \
#     --dataset /mnt/huawei/users/lfu/datasets/SWE-bench_Verified \
#     --repostructure_dir /mnt/huawei/users/lfu/codes/Agentless/artifacts/repo_structure/repo_structures/ \
#     --input_file results/bugfixer/Nemotron-Cascade-8B-Thinking/batch_output_step2.jsonl \
#     --save_dir results/bugfixer/Nemotron-Cascade-8B-Thinking/ \
#     --num_workers 32

