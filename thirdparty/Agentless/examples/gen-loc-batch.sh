export PYTHONPATH=$PYTHONPATH:$PWD
export PROJECT_FILE_LOC=/mnt/huawei/users/lfu/codes/Agentless/artifacts/repo_structure/repo_structures
INFER_OUTPUT=results/loc-inference/dsv32
MODEL="deepseek-ai/DeepSeek-V3.2/e5mbg25y3n"
# INFER_OUTPUT=results/inference/021-8b-swe
# MODEL=021-8B-CoT
python agentless/fl/localize.py --file_level --batch_gen \
                                  --output_folder ${INFER_OUTPUT} \
				  --model ${MODEL} \
				  --backend deepseek \
				  --temperature 1.0 \
				  --max_tokens 32768 \
				  --thinking_budget 32768 \
				  --enable_thinking \
                                  --batch_gen --dataset /mnt/huawei/users/lfu/datasets/SWE-bench_Verified
