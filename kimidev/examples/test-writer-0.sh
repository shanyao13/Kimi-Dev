#!/bin/bash

# Activate conda environment if needed
# source /path/to/conda/etc/profile.d/conda.sh
# conda activate swebench

# Set PYTHONPATH to include the project root
export PYTHONPATH="/mnt/huawei/users/swZheng/proDir/Kimi-Dev:$PYTHONPATH"
export MODEL_NAME="q3-8b"

# ### 2. Test Writer (`batch_rollout_testwriter.py`)

# # #### Stage 1: Generate Relevant File Requests
# python batch_rollout_testwriter.py \
#     gen_step1 \
#     --dataset /mnt/huawei/users/lfu/datasets/SWE-bench_Verified \
#     --repostructure_dir /mnt/huawei/users/lfu/codes/Agentless/artifacts/repo_structure/repo_structures/ \
#     --output_file results/testwriter/q3-8b/batch_requests_step1.jsonl \
#     --model_name "q3-8b" \
#     --temp 0.6 \
#     --passk 40 \
#     --max_tokens 45000 \
#     --thinking_budget 45000 \
#     --enable_thinking

# 这里think和no-think的比较？温度设置和思考模式设置。

# #### Stage 2: Generate Test Creation Requests
# python batch_rollout_testwriter.py \
#     proc_step1_gen_step2 \
#     --dataset /mnt/huawei/users/lfu/datasets/SWE-bench_Verified \
#     --repostructure_dir /mnt/huawei/users/lfu/codes/Agentless/artifacts/repo_structure/repo_structures/ \
#     --input_file results/testwriter/Nemotron-Cascade-8B-Thinking/splits-testwriter/batch_output_step1_part_0003.jsonl \
#     --output_file results/testwriter/Nemotron-Cascade-8B-Thinking/splits-testwriter/batch_requests_step2_part_0003.jsonl \
#     --model_name "Nemotron-Cascade-8B-Thinking" \
#     --temp 0.6 \
#     --max_tokens 45000 \
#     --thinking_budget 45000 \
#     --enable_thinking \
#     --num_workers 20 \
#     --convert_crlf

for input in /mnt/huawei/users/swZheng/proDir2026/Kimi-Dev/kimidev/examples/results/testwriter/${MODEL_NAME}/splits-testwriter/batch_output_step1_part_*.jsonl; do
    part=$(basename "$input" .jsonl | sed 's/.*part_//')
    output="/mnt/huawei/users/swZheng/proDir2026/Kimi-Dev/kimidev/examples/results/testwriter/${MODEL_NAME}/splits-testwriter/batch_requests_step2_part_${part}.jsonl"

    echo "Processing part ${part} ..."

    python batch_rollout_testwriter.py \
        proc_step1_gen_step2 \
        --dataset /mnt/huawei/users/lfu/datasets/SWE-bench_Verified \
        --repostructure_dir /mnt/huawei/users/lfu/codes/Agentless/artifacts/repo_structure/repo_structures/ \
        --input_file "$input" \
        --output_file "$output" \
        --model_name ${MODEL_NAME} \
        --temp 0.6 \
        --max_tokens 45000 \
        --thinking_budget 45000 \
        --enable_thinking \
        --num_workers 40 \
        --convert_crlf
done




# #### Stage 3: Process Tests and Save
# python batch_rollout_testwriter.py \
#     proc_step2 \
#     --dataset /mnt/huawei/users/lfu/datasets/SWE-bench_Verified \
#     --repostructure_dir /mnt/huawei/users/lfu/codes/Agentless/artifacts/repo_structure/repo_structures/ \
#     --input_file results/testwriter/Nemotron-Cascade-8B-Thinking/batch_output_step2.jsonl \
#     --step1_file results/testwriter/Nemotron-Cascade-8B-Thinking/batch_output_step1.jsonl \
#     --save_dir results/testwriter/Nemotron-Cascade-8B-Thinking \
#     --num_workers 8
#     # --passk 40
