import argparse
import asyncio
import json
import uuid
import aiohttp
from tqdm.asyncio import tqdm
import os

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
                    
                    if status_code != 200:
                         # Depending on how strict we want to be, we might put error info in "error" field
                         # But OpenAI batch format usually puts the http response in "response" even if error,
                         # unless the batch item itself failed processing.
                         # If the request failed at HTTP level, satisfied by status_code != 200.
                         pass

                    return json.dumps(output_data)

            except Exception as e:
                # Network error or other exception
                 return json.dumps({
                    "id": f"batch_req_{str(uuid.uuid4()).replace('-', '')[:20]}",
                    "custom_id": custom_id,
                    "response": None,
                    "error": {
                        "code": "internal_error",
                        "message": str(e)
                    }
                })

    except json.JSONDecodeError:
        return None

async def main():
    parser = argparse.ArgumentParser(description="Process OpenAI batch requests concurrently.")
    parser.add_argument("--input_file", type=str, required=True, help="Input .jsonl file with requests.")
    parser.add_argument("--output_file", type=str, required=True, help="Output .jsonl file for responses.")
    parser.add_argument("--url", type=str, default="http://localhost:30000/v1/chat/completions", help="Target API URL.")
    parser.add_argument("--concurrency", type=int, default=32, help="Number of concurrent requests.")
    parser.add_argument("--timeout", type=int, default=7200, help="Request timeout in seconds (default: 7200).")
    parser.add_argument("--model", type=str, help="Override model name in requests.")

    args = parser.parse_args()

    if not os.path.exists(args.input_file):
        print(f"Error: Input file {args.input_file} not found.")
        return

    # Count total lines for progress bar
    total_lines = 0
    with open(args.input_file, 'r') as f:
        for _ in f:
            total_lines += 1

    semaphore = asyncio.Semaphore(args.concurrency)
    
    print(f"Processing {total_lines} requests from {args.input_file}...")
    print(f"Target URL: {args.url}")
    print(f"Concurrency: {args.concurrency}")
    print(f"Timeout: {args.timeout} seconds")

    timeout = aiohttp.ClientTimeout(total=args.timeout)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        tasks = []
        with open(args.input_file, 'r') as f:
            for line in f:
                task = asyncio.create_task(process_request(session, line, args.url, semaphore, args.model))
                tasks.append(task)
        
        # Open output file
        with open(args.output_file, 'w') as f_out:
            for future in tqdm(asyncio.as_completed(tasks), total=total_lines):
                result = await future
                if result:
                    f_out.write(result + "\n")
                    f_out.flush() # Ensure it's written immediately-ish? Maybe not strictly necessary to flush every line.

    print(f"Done. Results written to {args.output_file}")

if __name__ == "__main__":
    asyncio.run(main())
