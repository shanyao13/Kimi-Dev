#!/bin/sh
#SBATCH --job-name=reproduction-tests
#SBATCH --cpus-per-task=64
#SBATCH --ntasks-per-node=1
#SBATCH --ntasks=1
#SBATCH --time=24:00:00
#SBATCH -p debug
#SBATCH --array 0-39

conda activate swebench

DATASET=../../datasets/SWE-bench_Verified

set -x

cd /mnt/huawei/users/lfu/codes/Kimi-Dev
export PYTHONPATH=$PYTHONPATH:${PWD}/thirdparty/Agentless/:$PWD
num=${SLURM_ARRAY_TASK_ID}
echo "Processing ${num}"
run_id_prefix=dsr1
python thirdparty/Agentless/agentless/test/run_reproduction_tests.py \
    --test_jsonl results/dsr1-testwriter/samples_processed-new/reproduction_tests.jsonl \
    --predictions_path="results/dsr1/samples_processed/output_${num}_processed.jsonl" \
    --dataset ${DATASET} \
    --run_id="${run_id_prefix}_reproduction_${num}" \
    --num_workers 10
