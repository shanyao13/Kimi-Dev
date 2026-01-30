
import json
import os
import argparse
import sys
from collections import defaultdict

# Add the repository root to sys.path to import agentless modules
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# Attempt to import repair functionality
try:
    from agentless.repair.repair import _post_process_multifile_repair, construct_topn_file_context, get_repo_structure, get_full_file_paths_and_classes_and_functions
    from agentless.util.api_requests import num_tokens_from_messages
except ImportError:
    # Fallback or strict error depending on environment. 
    # Since we are in the repo, it should work if PYTHONPATH is correct or via sys.path
    print("Warning: Could not import agentless modules. Post-processing might be limited.")
    _post_process_multifile_repair = None

class MockLogger:
    def info(self, msg):
        pass # print(f"[INFO] {msg}")
    def error(self, msg):
        print(f"[ERROR] {msg}")

def main():
    parser = argparse.ArgumentParser(description="Assemble OpenAI Batch API responses into output.jsonl files.")
    parser.add_argument("--batch_output", type=str, required=True, help="Input batch output .jsonl file.")
    parser.add_argument("--base_dir", type=str, default="artifacts/agentless_swebench_verified", help="Base directory containing repair samples.")
    parser.add_argument("--suffix", type=str, default="", help="Suffix for updated output files (e.g., '_assembled'). Leave empty to overwrite.")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.batch_output):
        print(f"Error: Batch output file {args.batch_output} does not exist.")
        return

    # Load batch responses
    # Structure: custom_id -> response_body
    responses_by_id = {}
    
    print(f"Loading batch responses from {args.batch_output}...")
    with open(args.batch_output, 'r') as f:
        for line in f:
            if not line.strip():
                continue
            try:
                data = json.loads(line)
                custom_id = data.get("custom_id")
                # OpenAI batch output structure:
                # { "id": "...", "custom_id": "...", "response": { "status_code": 200, "body": { ... } } }
                if custom_id:
                    responses_by_id[custom_id] = data
            except json.JSONDecodeError:
                pass
                
    print(f"Loaded {len(responses_by_id)} responses.")
    
    # Group by directory and instance_id
    # custom_id format: {directory_name}|{instance_id}|{sample_index}
    results_map = defaultdict(lambda: defaultdict(dict))
    
    for custom_id, data in responses_by_id.items():
        parts = custom_id.split("|")
        if len(parts) != 3:
            print(f"Warning: Invalid custom_id format: {custom_id}")
            continue
            
        dirname, instance_id, sample_idx = parts
        sample_idx = int(sample_idx)
        results_map[dirname][instance_id][sample_idx] = data

    logger = MockLogger()

    # Process each directory
    for dirname, instances in results_map.items():
        dir_path = os.path.join(args.base_dir, dirname)
        jsonl_path = os.path.join(dir_path, "output.jsonl")
        
        if not os.path.exists(jsonl_path):
            print(f"Warning: Original file not found at {jsonl_path}. Skipping assembly for {dirname}.")
            continue
            
        print(f"Updating {jsonl_path}...")
        
        updated_lines = []
        with open(jsonl_path, 'r') as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    original_data = json.loads(line)
                    instance_id = original_data.get("instance_id")
                    
                    if instance_id in instances:
                        # We have new data for this instance
                        samples = instances[instance_id]
                        
                        # Initialize lists if they are placeholders
                        # We expect 10 samples (idx 0-9)
                        # We need to reconstruct:
                        # - raw_output (list of strings)
                        # - all_generations (list of list of strings? No, looking at repair.py: all_generations.append(raw_output))
                        #   Wait, repair.py: "all_generations": [all_generations] --> list of list of strings.
                        #   Usually it seems to capture all attempts.
                        #   In repair.py:
                        #     all_generations.append(raw_output)
                        #     ...
                        #     "all_generations": [all_generations]
                        #   So it's nested [[gen0, gen1, ...]]
                        
                        new_raw_outputs = []
                        new_traj = []
                        new_edited_files_list = [] # correlates to file_names
                        new_prev_contents_list = []
                        new_try_count = []
                        
                        # Need to recover file_contents and file_loc_intervals to run post-processing
                        # But those are not stored in output.jsonl fully.
                        # The original output.jsonl has "original_file_content" which might be keys or content?
                        # repair.py writes: "original_file_content": content (which is a list of contents)
                        # But wait, logic in repair.py:
                        # file_contents is derived during execution.
                        # If we want to post-process, we need the file contents.
                        # For now, maybe we just store the raw outputs and let a subsequent step do the diff generation?
                        # OR, we try to grab the content if available.
                        # The prompt contains the file content! We can extract it from the prompt if needed, 
                        # or just store raw output and let the user run a post-process script (repair.py has a --post_process flag).
                        
                        # The user asked to "assemble back".
                        # If I assume the user will run post-processing later, I just need to fill in `all_generations` and `traj`.
                        # However, repair.py --post_process expects `all_generations` to be populated.
                        
                        # Let's populate:
                        # traj: list of dicts with response, usage, prompt
                        # all_generations: [[gen0, gen1, ...]] 
                        # raw_output: list of strings (the effective ones?)
                        
                        # Sort samples by index
                        sorted_indices = sorted(samples.keys())
                        
                        collected_generations = []
                        collected_traj = []
                        
                        # Preserve original prompt if possible
                        original_prompt = ""
                        if original_data.get("traj"):
                            original_prompt = original_data["traj"][0].get("prompt", "")
                        
                        for idx in sorted_indices:
                            resp_data = samples[idx]
                            response_obj = resp_data.get("response", {})
                            body = response_obj.get("body", {})
                            choices = body.get("choices", [])
                            
                            content = ""
                            if choices:
                                content = choices[0].get("message", {}).get("content", "")
                            
                            usage = body.get("usage", {})
                            
                            traj_item = {
                                "response": content,
                                "usage": usage,
                                "prompt": original_prompt # We might not have the full prompt here if we don't reload it, but we should try.
                            }
                            collected_traj.append(traj_item)
                            collected_generations.append(content)
                        
                        # Update the data
                        # Note: repair.py structure for 'all_generations' is [[gen0, gen1...]]
                        original_data["all_generations"] = [collected_generations]
                        original_data["traj"] = collected_traj
                        # raw_output in repair.py seems to be the list of outputs too?
                        # "raw_output": raw_outputs (list of strings)
                        original_data["raw_output"] = collected_generations 
                        
                        # We also need to reset 'try_count' maybe?
                        original_data["try_count"] = list(range(1, len(collected_generations) + 1))
                        
                        updated_lines.append(json.dumps(original_data))
                    else:
                        updated_lines.append(line.strip())
                        
                except Exception as e:
                    print(f"Error processing line in {jsonl_path}: {e}")
                    updated_lines.append(line.strip())
        
        # Write back
        out_path = jsonl_path
        if args.suffix:
            out_path = jsonl_path.replace(".jsonl", f"{args.suffix}.jsonl")
            
        with open(out_path, 'w') as f:
            for l in updated_lines:
                f.write(l + "\n")
        print(f"Wrote to {out_path}")

if __name__ == "__main__":
    main()
