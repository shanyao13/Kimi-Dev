export PYTHONPATH=$PYTHONPATH:$PWD
export PROJECT_FILE_LOC=/mnt/huawei/users/lfu/codes/Agentless/artifacts/repo_structure/repo_structures
# INFER_OUTPUT=results/inference/dsr1
# MODEL="deepseek-ai/DeepSeek-R1/e5mbg25y3n"
INFER_OUTPUT=results/inference/021-8b-swe
MODEL=021-8B-CoT
python agentless/repair/repair.py --loc_file results/oracle_loc/loc_outputs.jsonl \
                                  --output_folder ${INFER_OUTPUT} \
                                  --loc_interval \
                                  --top_n=none \
                                  --context_window=10 \
                                  --max_samples 11  \
				  --skip_greedy \
                                  --cot \
                                  --diff_format \
				  --model ${MODEL} \
				  --temperature 0.6 \
				  --max_tokens 32768 \
				  --thinking_budget 32768 \
				  --enable_thinking \
                                  --batch_gen --dataset /mnt/huawei/users/lfu/datasets/SWE-bench_Verified
