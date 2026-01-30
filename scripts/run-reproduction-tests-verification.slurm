#!/bin/sh
#SBATCH --job-name=reproduction
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
python thirdparty/Agentless/agentless/test/run_reproduction_tests.py \
    --run_id="reproduction_test_generation_filter_sample_${num}" \
    --test_jsonl="./kimidev/examples/results/testwriter/Nemotron-Cascade-8B-Thinking/output_${num}_processed_reproduction_test.jsonl" \
    --dataset ${DATASET} \
    --num_workers 10 \
    --testing



#后面跑：majority voting to select one reproduction test per issue