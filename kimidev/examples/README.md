# Batch Rollout Scripts

This directory contains scripts for running the Agentless workflow using the OpenAI Batch API (or compatible batch endpoints). The workflow is split into three distinct stages to allow for async processing.

## Available Scripts

*   `batch_rollout_bugfixer.py`: For generating bug fixes (patches).
*   `batch_rollout_testwriter.py`: For generating reproduction test scripts.

## Workflow Overview

Both scripts follow the same 3-stage pattern:

1.  **Stage 1 (`gen_step1`)**:
    *   **Goal**: Generate prompts to identify relevant files.
    *   **Input**: Dataset (e.g., `princeton-nlp/SWE-bench_Lite`).
    *   **Output**: A JSONL file containing batch requests (e.g., `batch_requests_step1.jsonl`).

2.  **Stage 2 (`proc_step1_gen_step2`)**:
    *   **Goal**: Process the relevant files found in Stage 1 and generate prompts for the main task (fix or test generation).
    *   **Input**: The output JSONL from the Batch API execution of Stage 1 (e.g., `batch_output_step1.jsonl`).
    *   **Output**: A JSONL file containing batch requests for Step 2 (e.g., `batch_requests_step2.jsonl`).

3.  **Stage 3 (`proc_step2`)**:
    *   **Goal**: Process the final responses from Stage 2 and save the results (patches or test scripts).
    *   **Input**: The output JSONL from the Batch API execution of Stage 2 (e.g., `batch_output_step2.jsonl`).
    *   **Output**: Saved files in the `samples/` directory and aggregated logs.

---

## Usage Examples

### 1. Bug Fixer (`batch_rollout_bugfixer.py`)

#### Stage 1: Generate Relevant File Requests
```bash
python batch_rollout_bugfixer.py \
    --stage gen_step1 \
    --dataset princeton-nlp/SWE-bench_Lite \
    --repostructure_dir ./repostructures \
    --output_file results/dsr1-bugfixer/batch_requests_step1.jsonl \
    --model_name "deepseek-ai/DeepSeek-V3.2" \
    --temp 1.0 \
    --passk 1
```
*   **Result**: `results/dsr1-bugfixer/batch_requests_step1.jsonl`

*(User submits this file to Batch API and gets `batch_output_step1.jsonl`)*

#### Stage 2: Generate Repair Requests
```bash
python batch_rollout_bugfixer.py \
    --stage proc_step1_gen_step2 \
    --dataset princeton-nlp/SWE-bench_Lite \
    --repostructure_dir ./repostructures \
    --input_file results/dsr1-bugfixer/batch_output_step1.jsonl \
    --output_file results/dsr1-bugfixer/batch_requests_step2.jsonl \
    --model_name "deepseek-ai/DeepSeek-V3.2" \
    --temp 1.0 \
    --passk 1 \
    --num_workers 32
```
*   **Result**: `results/dsr1-bugfixer/batch_requests_step2.jsonl`

*(User submits this file to Batch API and gets `batch_output_step2.jsonl`)*

#### Stage 3: Process Repairs and Save
```bash
python batch_rollout_bugfixer.py \
    --stage proc_step2 \
    --dataset princeton-nlp/SWE-bench_Lite \
    --repostructure_dir ./repostructures \
    --input_file results/dsr1-bugfixer/batch_output_step2.jsonl \
    --save_dir results/dsr1-bugfixer \
    --passk 1 \
    --num_workers 32
```
*   **Result**: Patches saved in `results/dsr1-bugfixer/samples/` and `output.jsonl`.

---

### 2. Test Writer (`batch_rollout_testwriter.py`)

#### Stage 1: Generate Relevant File Requests
```bash
python batch_rollout_testwriter.py \
    --stage gen_step1 \
    --dataset princeton-nlp/SWE-bench_Lite \
    --repostructure_dir ./repostructures \
    --output_file results/dsr1-testwriter/batch_requests_step1.jsonl \
    --model_name "deepseek-ai/DeepSeek-V3.2" \
    --temp 1.0 \
    --passk 1
```

#### Stage 2: Generate Test Creation Requests
```bash
python batch_rollout_testwriter.py \
    --stage proc_step1_gen_step2 \
    --dataset princeton-nlp/SWE-bench_Lite \
    --repostructure_dir ./repostructures \
    --input_file results/dsr1-testwriter/batch_output_step1.jsonl \
    --output_file results/dsr1-testwriter/batch_requests_step2.jsonl \
    --model_name "deepseek-ai/DeepSeek-V3.2" \
    --temp 1.0 \
    --passk 1
```

#### Stage 3: Process Tests and Save
```bash
python batch_rollout_testwriter.py \
    --stage proc_step2 \
    --dataset princeton-nlp/SWE-bench_Lite \
    --repostructure_dir ./repostructures \
    --input_file results/dsr1-testwriter/batch_output_step2.jsonl \
    --step1_file results/dsr1-testwriter/batch_output_step1.jsonl \
    --save_dir results/dsr1-testwriter \
    --passk 1
```
*   **Result**: Reproduction scripts saved in `results/dsr1-testwriter/samples/` and `failures.jsonl`.

---

## Common Arguments

*   `--save_dir`: Directory to save generated JSONL files and output samples.
*   `--dataset`: HuggingFace dataset name or path.
*   `--passk`: Number of samples (passes) being generated. E.g., if you generated 8 samples per problem in the Batch API, set this to 8. The scripts handle `__passN` suffixes in `custom_id` automatically.
*   `--num_workers`: Number of parallel workers for processing. Recommended for Stage 2 and 3 with large datasets processing.
*   `--start_file`, `--end_file`: Process a specific range of `custom_id`s (sorted stringwise).
*   `--enable_thinking`, `--thinking_budget`: For thinking models like DeepSeek-R1.
*   `--convert_crlf`: Normalize problem statement line endings.
