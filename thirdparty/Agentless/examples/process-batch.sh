export PROJECT_FILE_LOC=/mnt/huawei/users/lfu/codes/Agentless/artifacts/repo_structure/repo_structures
# BATCH_OUTPUT=./results/oracle_batch/repair/responses-repair.jsonl
# OUTPUT_FOLDER=results/oracle_batch/repair-2
# MODEL="deepseek-ai/DeepSeek-V3.2/e5mbg25y3n"
# BATCH_OUTPUT=./results/inference/dsr1/responses-repair.jsonl
# OUTPUT_FOLDER=results/oracle_batch/repair-dsr1
# MODEL="deepseek-ai/DeepSeek-R1/e5mbg25y3n"
BATCH_OUTPUT=results/inference/021-8b-swe/responses-repair.jsonl
OUTPUT_FOLDER=results/oracle_batch/repair-021-8b-swe
MODEL="021-8B-CoT-SWE"
python agentless/repair/repair.py --loc_file results/oracle_loc/loc_outputs.jsonl \
                                  --output_folder ${OUTPUT_FOLDER} \
                                  --loc_interval \
                                  --top_n=none \
                                  --context_window=10 \
                                  --max_samples 11  \
				  --skip_greedy \
                                  --cot \
                                  --diff_format \
				  --model "${MODEL}" \
				  --temperature 1.0 \
				  --max_tokens 32768 \
				  --thinking_budget 32768 \
				  --enable_thinking \
				  --num_threads 20 \
				  --process_batch --batch_output ${BATCH_OUTPUT} --gen_and_process \
				  --dataset /mnt/huawei/users/lfu/datasets/SWE-bench_Verified
