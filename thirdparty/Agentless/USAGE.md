# Batch Repair Usage

This document details how to use the batch generation and processing features in `agentless/repair/repair.py`. These features allow you to decouple the prompt generation from the inference process, enabling high-throughput parallel inference using engines like SGLang or other OpenAI-compatible endpoints.

The workflow consists of three steps:
1.  **Generate Batch Requests**: Create a JSONL file containing OpenAI-compatible chat completion requests.
2.  **Execute Batch Requests**: Run the requests against an LLM endpoint (e.g., SGLang) and save the responses.
3.  **Process Batch Responses**: Feed the responses back into Agentless to generate the final repair patches.

## Step 1: Generate Batch Requests (`--batch_gen`)

Use the `--batch_gen` flag with `agentless/repair/repair.py` to generate the request payloads. Instead of calling the API immediately, this will append request JSON objects to the file specified by `--output_file` (or `args.output_file` logic).

**Note**: Ensure your `PYTHONPATH` includes the current directory.

### Example Command

```bash
export PYTHONPATH=$PYTHONPATH:$PWD
export PROJECT_FILE_LOC=/path/to/repo_structures  # Update this path
INFER_OUTPUT=results/inference/my_experiment
MODEL="deepseek-ai/DeepSeek-V3"

mkdir -p ${INFER_OUTPUT}

python agentless/repair/repair.py \
    --loc_file results/oracle_loc/loc_outputs.jsonl \
    --output_folder ${INFER_OUTPUT} \
    --loc_interval \
    --top_n=none \
    --context_window=10 \
    --max_samples 11 \
    --skip_greedy \
    --cot \
    --diff_format \
    --model ${MODEL} \
    --temperature 0.6 \
    --max_tokens 32768 \
    --thinking_budget 32768 \
    --enable_thinking \
    --batch_gen \
    --dataset /path/to/SWE-bench_Verified  # Update this path
```

**Key Arguments:**
*   `--batch_gen`: Enables batch generation mode. Requests are written to the output file instead of being executed.
*   `--output_folder`: Directory where the request file (and logs) will be stored.
*   `--skip_greedy`: (Optional) Skip the greedy decoding sample if you only want temperature-sampled requests.

After this step, you will have a file (e.g., in `${INFER_OUTPUT}`) containing the JSONL requests.

## Step 2: Execute Batch Requests

You can use the provided helper scripts to process the generated request file. `scripts/serve_and_process_batch_requests.py` can optionally launch an SGLang server and process the requests concurrently.

### Example Command

```bash
# Input file generated in Step 1
INPUT_FILE=results/inference/021-8b-swe/requests-repair.jsonl
# Output file for model responses
BATCH_OUTPUT=results/inference/${MODEL_NAME}/responses-repair.jsonl
MODEL_PATH=/path/to/model/weights

# Option A: With existing server
python scripts/process_batch_requests.py \
    --input_file ${INPUT_FILE} \
    --output_file ${BATCH_OUTPUT} \
    --url http://localhost:30000/v1/chat/completions \
    --concurrency 64

# Option B: Launch SGLang server and process
# Requires sglang installed and gpu available
mkdir -p results/inference/${MODEL_NAME}
python scripts/serve_and_process_batch_requests.py \
    --input_file ${INPUT_FILE} \
    --output_file ${BATCH_OUTPUT} \
    --url http://127.0.0.1:8000/v1/chat/completions  \
    --concurrency 32 \
    --timeout 3600 \
    --serve \
    --tp-size 1 --dp-size 8 --model-path ${MODEL_PATH} --reasoning-parser deepseek-r1
```

**Helper Scripts:**
*   `scripts/process_batch_requests.py`: Sends requests to an existing URL.
*   `scripts/serve_and_process_batch_requests.py`: Can launch SGLang (`--serve`) and then process requests. It manages the server process for you.

## Step 3: Process Batch Responses (`--process_batch`)

Once you have the responses in a JSONL file, use `--process_batch` to generate the final patches. You must provide the response file path via `--batch_output`.

### Example Command

```bash
export PYTHONPATH=$PYTHONPATH:$PWD
export PROJECT_FILE_LOC=/path/to/repo_structures # Update this path
BATCH_OUTPUT=results/inference/my_experiment/responses.jsonl
OUTPUT_FOLDER=results/inference/my_experiment_final
MODEL="deepseek-ai/DeepSeek-V3"

python agentless/repair/repair.py \
    --loc_file results/oracle_loc/loc_outputs.jsonl \
    --output_folder ${OUTPUT_FOLDER} \
    --loc_interval \
    --top_n=none \
    --context_window=10 \
    --max_samples 11 \
    --skip_greedy \
    --cot \
    --diff_format \
    --model "${MODEL}" \
    --temperature 1.0 \
    --max_tokens 32768 \
    --thinking_budget 32768 \
    --enable_thinking \
    --num_threads 20 \
    --process_batch \
    --batch_output ${BATCH_OUTPUT} \
    --dataset /path/to/SWE-bench_Verified # Update this path
```

**Key Arguments:**
*   `--process_batch`: Tells the script to read from a batch output file instead of querying the model.
*   `--batch_output`: Path to the JSONL file containing the model responses (generated in Step 2).
*   `--num_threads`: Controls the parallelism for post-processing the responses (patch extraction, etc.).

This will generate the final repair results and logs in `${OUTPUT_FOLDER}`.
