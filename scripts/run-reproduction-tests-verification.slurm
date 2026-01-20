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

DATASET=../../datasets/SWE-bench_Verified

set -x

cd /mnt/huawei/users/lfu/codes/Kimi-Dev
export PYTHONPATH=$PYTHONPATH:${PWD}/thirdparty/Agentless/:$PWD
num=${SLURM_ARRAY_TASK_ID}
echo "Processing ${num}"
python thirdparty/Agentless/agentless/test/run_reproduction_tests.py --run_id="reproduction_test_generation_filter_sample_${num}" \
    --test_jsonl="results/dsr1-testwriter/samples_processed-new/output_${num}_processed_reproduction_test.jsonl" \
    --dataset ${DATASET} \
    --num_workers 10 \
    --testing
