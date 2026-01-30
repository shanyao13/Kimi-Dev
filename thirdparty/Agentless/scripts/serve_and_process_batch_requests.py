import argparse
import asyncio
import json
import uuid
import aiohttp
from tqdm import tqdm as tqdm_std
from tqdm.asyncio import tqdm
import os
import sys
import signal
import time

async def process_request(session, line, url, semaphore, model=None):
    if not line.strip():
        return None

    try:
        request_data = json.loads(line)
        custom_id = request_data.get("custom_id")
        body = request_data.get("body", {})
        
        # Override model if specified
        if model:
            body["model"] = model
            
        async with semaphore:
            try:
                async with session.post(url, json=body) as response:
                    status_code = response.status
                    response_json = await response.json()
                    
                    # Construct the output object in OpenAI Batch format
                    output_data = {
                        "id": f"batch_req_{str(uuid.uuid4()).replace('-', '')[:20]}",
                        "custom_id": custom_id,
                        "response": {
                            "status_code": status_code,
                            "request_id": response.headers.get("x-request-id", ""),
                            "body": response_json
                        },
                        "error": None
                    }
                    
                    return json.dumps(output_data)

            except Exception as e:
                # Network error or other exception
                 return json.dumps({
                    "id": f"batch_req_{str(uuid.uuid4()).replace('-', '')[:20]}",
                    "custom_id": custom_id,
                    "response": {
                        "status_code": 500,
                        "request_id": "",
                        "body": {}
                    },
                    "error": {
                        "code": "internal_error",
                        "message": str(e)
                    }
                })

    except json.JSONDecodeError:
        return None

async def log_subprocess_output(stream, prefix="[SGLang]"):
    """Reads lines from stream and prints them using tqdm.write to keep it above original tqdm bars."""
    while True:
        line = await stream.readline()
        if not line:
            break
        msg = line.decode().strip()
        if msg:
            tqdm_std.write(f"{prefix} {msg}")

async def wait_for_server(url, timeout=300):
    """Wait for the SGLang server to be ready by polling the /v1/models endpoint."""
    start_time = time.time()
    tqdm_std.write(f"Waiting for SGLang server at {url}...")
    
    # Extract base URL (e.g., http://localhost:30000)
    base_url = "/".join(url.split("/")[:3])
    health_url = f"{base_url}/v1/models"
    
    async with aiohttp.ClientSession() as session:
        while time.time() - start_time < timeout:
            try:
                async with session.get(health_url) as response:
                    if response.status == 200:
                        tqdm_std.write("SGLang server is ready!")
                        return True
            except Exception:
                pass
            await asyncio.sleep(2)
    
    tqdm_std.write("Timeout waiting for SGLang server.")
    return False

async def main():
    parser = argparse.ArgumentParser(description="Process OpenAI batch requests concurrently.")
    parser.add_argument("--input_file", type=str, required=True, help="Input .jsonl file with requests.")
    parser.add_argument("--output_file", type=str, required=True, help="Output .jsonl file for responses.")
    parser.add_argument("--url", type=str, default="http://localhost:30000/v1/chat/completions", help="Target API URL.")
    parser.add_argument("--concurrency", type=int, default=32, help="Number of concurrent requests.")
    parser.add_argument("--timeout", type=int, default=7200, help="Request timeout in seconds (default: 7200).")
    parser.add_argument("--model", type=str, help="Override model name in requests.")
    parser.add_argument("--serve", action="store_true", help="Launch SGLang server using remaining arguments.")
    
    # Capture all remaining arguments for sglang
    args, sglang_args = parser.parse_known_args()

    if not os.path.exists(args.input_file):
        print(f"Error: Input file {args.input_file} not found.")
        return

    # Count total lines for progress bar
    total_lines = 0
    with open(args.input_file, 'r') as f:
        for _ in f:
            total_lines += 1

    semaphore = asyncio.Semaphore(args.concurrency)
    
    current_url = args.url
    sglang_proc = None
    log_task = None
    
    host = "127.0.0.1"
    port = 30000

    try:
        if args.serve:
            # Find a free port starting from 30000
            import socket
            def find_free_port(start_port):
                p = start_port
                while p < 65535:
                    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                        if s.connect_ex((host, p)) != 0:
                            return p
                    p += 1
                return start_port

            port = find_free_port(30000)
            # Construct sglang launch command
            launch_cmd = [sys.executable, "-m", "sglang.launch_server", "--host", host, "--port", str(port)] + sglang_args
            
            tqdm_std.write(f"Launching SGLang on {host}:{port} with args: {' '.join(sglang_args)}")
            sglang_proc = await asyncio.create_subprocess_exec(
                *launch_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT
            )
            
            # Start logging task
            log_task = asyncio.create_task(log_subprocess_output(sglang_proc.stdout))
            
            # Update current_url
            current_url = f"http://{host}:{port}/v1/chat/completions"
            
            # Wait for server to be ready
            ready = await wait_for_server(current_url)
            if not ready:
                print("Failed to start SGLang server.")
                return

        print(f"Processing {total_lines} requests from {args.input_file}...")
        print(f"Target URL: {current_url}")
        print(f"Concurrency: {args.concurrency}")
        print(f"Timeout: {args.timeout} seconds")

        # Check for existing progress
        processed_ids = set()
        if os.path.exists(args.output_file):
            print(f"Output file {args.output_file} exists. Checking for resumable progress...")
            with open(args.output_file, 'r') as f_out_read:
                for line in f_out_read:
                    if line.strip():
                        try:
                            data = json.loads(line)
                            if "custom_id" in data:
                                # Check if successful or permanent failure
                                is_success = False
                                is_permanent_fail = False
                                
                                # Check response status
                                if data.get("response") and isinstance(data["response"], dict):
                                    status = data["response"].get("status_code")
                                    if status == 200:
                                        is_success = True
                                    else:
                                        # Check body for specific error message
                                        body_str = json.dumps(data["response"].get("body", {}))
                                        if "maximum context length" in body_str:
                                            is_permanent_fail = True
                                
                                # Check top-level error (network/client errors)
                                if data.get("error"):
                                    err_msg = data["error"].get("message", "")
                                    if "maximum context length" in str(err_msg):
                                        is_permanent_fail = True
                                
                                if is_success or is_permanent_fail:
                                    processed_ids.add(data["custom_id"])
                                    
                        except json.JSONDecodeError:
                            pass
            print(f"Found {len(processed_ids)} already processed requests (success or permanent fail). Resuming...")

        timeout = aiohttp.ClientTimeout(total=args.timeout)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            tasks = []
            skipped_count = 0
            with open(args.input_file, 'r') as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        req_data = json.loads(line)
                        c_id = req_data.get("custom_id")
                        if c_id in processed_ids:
                            skipped_count += 1
                            continue
                    except json.JSONDecodeError:
                        continue

                    task = asyncio.create_task(process_request(session, line, current_url, semaphore, args.model))
                    tasks.append(task)
            
            print(f"Skipping {skipped_count} requests already processed.")
            print(f"Queueing {len(tasks)} new requests.")
            
            # Open output file in append mode
            if tasks:
                with open(args.output_file, 'a') as f_out:
                    async for future in tqdm(asyncio.as_completed(tasks), total=len(tasks)):
                        result = await future
                        if result:
                            f_out.write(result + "\n")
                            f_out.flush() 

        print(f"Done. Results appended to {args.output_file}")

    finally:
        if log_task:
            log_task.cancel()
        if sglang_proc:
            tqdm_std.write("Shutting down SGLang server...")
            try:
                sglang_proc.terminate()
                await sglang_proc.wait()
            except Exception:
                sglang_proc.kill()
            tqdm_std.write("SGLang server shut down.")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass

