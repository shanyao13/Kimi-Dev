
import glob
import json
import os
import argparse
import re

def main():
    parser = argparse.ArgumentParser(description="Generate OpenAI Batch API requests from output.jsonl files.")
    parser.add_argument("--base_dir", type=str, default="artifacts/agentless_swebench_verified", help="Base directory containing repair samples.")
    parser.add_argument("--output_file", type=str, default="batch_requests.jsonl", help="Output file for batch requests.")
    parser.add_argument("--model", type=str, default="gpt-4o-2024-05-13", help="Model to use for requests.")
    parser.add_argument("--temperature", type=float, default=0.8, help="Temperature for sampling requests (default: 0.8).")
    parser.add_argument("--max_tokens", type=int, default=1024, help="Max tokens for requests (default: 1024).")
    parser.add_argument("--enable_thinking", action="store_true", help="Enable thinking parameter in requests.")
    
    args = parser.parse_args()
    
    # regex to match repair_sample_N where N is a number
    # explicitly ignoring _dsv32, _mock etc based on requirement
    sample_dir_pattern = re.compile(r"repair_sample_\d+$")
    
    all_requests = []
    
    if not os.path.exists(args.base_dir):
        print(f"Error: Base directory {args.base_dir} does not exist.")
        return

    dirs = sorted([d for d in os.listdir(args.base_dir) if os.path.isdir(os.path.join(args.base_dir, d))])
    
    processed_count = 0
    
    for d in dirs:
        if not sample_dir_pattern.match(d):
            print(f"Skipping directory: {d}")
            continue
            
        jsonl_path = os.path.join(args.base_dir, d, "output.jsonl")
        if not os.path.exists(jsonl_path):
            print(f"Warning: No output.jsonl found in {d}")
            continue
            
        print(f"Processing {d}...")
        
        with open(jsonl_path, 'r') as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    data = json.loads(line)
                    instance_id = data.get("instance_id")
                    traj = data.get("traj", [])
                    
                    if not instance_id or not traj:
                        print(f"Warning: Invalid data in {d}/output.jsonl for line: {line[:50]}...")
                        continue
                        
                    # We expect the prompt to be in the first trajectory item (the failed greedy attempt usually)
                    # OR we can reconstruct it. The user task implies extracting prompts.
                    # Looking at repair.py, 'traj' is a list of dicts with 'prompt' key.
                    
                    prompt = None
                    for t in traj:
                        if "prompt" in t:
                            prompt = t["prompt"]
                            break
                    
                    if not prompt:
                        print(f"Warning: No prompt found for {instance_id} in {d}")
                        continue
                        
                    # Generate 10 requests
                    # 1 greedy (temp=0)
                    # 9 samples (temp=0.8)
                    
                    # Request 0: Greedy
                    custom_id_0 = f"{d}|{instance_id}|0"
                    body_0 = {
                        "model": args.model,
                        "messages": [{"role": "user", "content": prompt}],
                        "max_tokens": args.max_tokens,
                        "temperature": 0
                    }
                    if args.enable_thinking:
                         body_0["enable_thinking"] = True
 
                    req_0 = {
                        "custom_id": custom_id_0,
                        "method": "POST",
                        "url": "/v1/chat/completions",
                        "body": body_0
                    }
                    all_requests.append(req_0)
                    
                    # Requests 1-9: Samples
                    for i in range(1, 10):
                        custom_id = f"{d}|{instance_id}|{i}"
                        body_sample = {
                            "model": args.model,
                            "messages": [{"role": "user", "content": prompt}],
                            "max_tokens": args.max_tokens,
                            "temperature": args.temperature
                        }
                        if args.enable_thinking:
                            body_sample["enable_thinking"] = True
 
                        req = {
                            "custom_id": custom_id,
                            "method": "POST",
                            "url": "/v1/chat/completions",
                            "body": body_sample
                        }
                        all_requests.append(req)
                        
                    processed_count += 1
                    
                except json.JSONDecodeError:
                    print(f"Error decoding JSON in {d}/output.jsonl")

    print(f"Total instances processed: {processed_count}")
    print(f"Total requests generated: {len(all_requests)}")
    
    with open(args.output_file, 'w') as f:
        for req in all_requests:
            f.write(json.dumps(req) + "\n")
    
    print(f"Batch requests written to {args.output_file}")

if __name__ == "__main__":
    main()
