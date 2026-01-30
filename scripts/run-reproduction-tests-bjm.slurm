#!/bin/sh
#SBATCH --job-name=reproduction-Nemotron-agentlesstest-kimi-bugfixer-shortest-bjm
#SBATCH --cpus-per-task=64
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --ntasks=1
#SBATCH --time=24:00:00
#SBATCH -p debug
#SBATCH --array 0-39

conda activate swebench

DATASET=/mnt/huawei/users/lfu/datasets/SWE-bench_Verified

set -x

cd /mnt/huawei/users/swZheng/proDir2026/Kimi-Dev
export PYTHONPATH=$PYTHONPATH:${PWD}/thirdparty/Agentless/:$PWD
num=${SLURM_ARRAY_TASK_ID}
echo "Processing ${num}"
run_id_prefix=Nemotron-agentlesstest-kimi-bugfixer-shortest-bjm
python thirdparty/Agentless/agentless/test/run_reproduction_tests.py \
    --test_jsonl /mnt/huawei/users/lfu/codes/Agentless/artifacts/agentless_swebench_verified/reproduction_test_samples/reproduction_tests.jsonl \
    --predictions_path="./kimidev/examples/results/bugfixer/Nemotron-Cascade-8B-Thinking-agentless-reproduction/output_${num}_processed.jsonl" \
    --dataset ${DATASET} \
    --run_id="${run_id_prefix}_reproduction_${num}" \
    --num_workers 10
