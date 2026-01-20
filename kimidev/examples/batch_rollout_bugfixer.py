import json
import os
import argparse
from tqdm import tqdm
from datasets import load_dataset
from kimidev.agentlessnano.utils import *
from kimidev.agentlessnano.post_process import generate_model_patch_difflib
import functools
import concurrent.futures

@functools.lru_cache(maxsize=50)
def get_repo_structure(instance_id, repostructure_dir):
    return search_instance_id_and_extract_structure(instance_id, repostructure_dir)

def get_indent(line):
    return len(line) - len(line.lstrip())

def adjust_indentation(replace_block, indent_diff):
    lines = replace_block.splitlines(keepends=True)
    adjusted_lines = []
    for line in lines:
        if not line.strip():
            adjusted_lines.append(line)
            continue
            
        current_indent = get_indent(line)
        new_indent = current_indent + indent_diff
        if new_indent < 0: 
            new_indent = 0 # Safety clamp
            
        adjusted_lines.append(" " * new_indent + line.lstrip())
    
    return "".join(adjusted_lines)

def apply_patches_robust(original_content, search_block, replace_block):

    # 3. Soft match (ignore leading/trailing whitespace per line)
    content_lines = original_content.splitlines(keepends=True)
    search_lines = search_block.splitlines(keepends=True)
    
    n_search = len(search_lines)
    if n_search == 0: return original_content
    
    for i in range(len(content_lines) - n_search + 1):
        window = content_lines[i:i+n_search]
        match = True
        for wa, wb in zip(window, search_lines):
            if wa.strip() != wb.strip():
                match = False
                break
        
        if match:
            # Found match!
            actual_first_line = content_lines[i]
            # Use the first non-empty line in search for reliable diff calculation
            # If all are empty, diff is 0
            search_first_line_idx = 0
            for idx, line in enumerate(search_lines):
                if line.strip():
                    search_first_line_idx = idx
                    break
            
            search_first_line = search_lines[search_first_line_idx]
            
            # Note: actual_first_line matches search_first_line in content (at i + search_first_line_idx)
            # wait, i points to start of block.
            actual_line_for_indent = content_lines[i + search_first_line_idx]
            
            indent_actual = get_indent(actual_line_for_indent)
            indent_search = get_indent(search_first_line)
            diff = indent_actual - indent_search
            
            # Adjust replace block
            adjusted_replace = adjust_indentation(replace_block, diff)

            pre = "".join(content_lines[:i])
            post = "".join(content_lines[i+n_search:])
            return pre + adjusted_replace + post
            
    return original_content

def generate_robust_diff(structure, search_replace_text):
    import re
    import difflib
    
    # Parse SEARCH/REPLACE blocks
    pattern = r'(?:### )?([^\n]+)\n<{4,} SEARCH\n(.*?)\n={4,}\n(.*?)(?:\n)?>{4,} REPLACE'
    matches = re.findall(pattern, search_replace_text, re.DOTALL)
    
    file_level_diff = ""
    edited_files_list = []
    original_contents_list = []
    new_contents_list = []
    
    # Group by file
    file_edits = {}
    for file_name, search_str, replace_str in matches:
        if file_name.strip() == "": continue
        
        # Ensure replace_str ends with newline if not empty, 
        # because we are doing line-based replacement
        if replace_str and not replace_str.endswith('\n'):
            replace_str += '\n'
            
        file_name = file_name.strip()
        if file_name not in file_edits:
            file_edits[file_name] = []
        file_edits[file_name].append((search_str, replace_str))
        
    # Apply edits
    files, _, _ = get_full_file_paths_and_classes_and_functions(structure)
    
    for file_name, edits in file_edits.items():
        # Correct path
        full_path = correct_file_path_in_structure(file_name, structure)
        
        # Get content
        # Note: get_repo_files expects a list
        file_content_dict = get_repo_files(structure, [full_path])
        if full_path not in file_content_dict:
            continue
            
        original = file_content_dict[full_path]
        modified = original
        
        for search_str, replace_str in edits:
            modified = apply_patches_robust(modified, search_str, replace_str)
            
        if modified == original:
            continue
            
        # Generate diff
        diff = difflib.unified_diff(
            original.splitlines(keepends=True),
            modified.splitlines(keepends=True),
            fromfile=f"a/{file_name}",
            tofile=f"b/{file_name}"
        )
        
        # Format diff for git apply (unified_diff return generator)
        # We need to manually construct the header if we want strict git format or just use the output?
        # difflib.unified_diff produces:
        # --- a/file
        # +++ b/file
        # @@ ... @@
        #
        # git apply expects:
        # diff --git a/file b/file
        # index ...
        # --- a/file
        # +++ b/file
        
        diff_lines = list(diff)
        if not diff_lines:
            continue
            
        header = [f"diff --git a/{file_name} b/{file_name}\n"]
        # We can skip index line or fake it
        # header.append("index 0000000..0000000 100644\n") 
        # Actually git apply is happy with just diff --git + --- + +++
        
        # But wait, difflib output usually starts with --- and +++
        # Let's check what generate_model_patch_difflib returns.
        # It manually constructs "diff --git..." then iterates lines.
        
        now_diff_content = f"diff --git a/{file_name} b/{file_name}\n"
        for line in diff_lines:
            if line.startswith('--- '):
                now_diff_content += f"--- a/{file_name}\n"
            elif line.startswith('+++ '):
                now_diff_content += f"+++ b/{file_name}\n"
            else:
                if not line.endswith('\n'):
                    line += '\n'
                now_diff_content += line
        
        file_level_diff += now_diff_content
        
        file_level_diff += now_diff_content
        
        # Capture content for edited files
        edited_files_list.append(file_name)
        original_contents_list.append(original)
        new_contents_list.append(modified)
        
    return file_level_diff, edited_files_list, original_contents_list, new_contents_list

def make_request_body(custom_id, model_name, messages, max_tokens, temperature, enable_thinking=False, thinking_budget=32768):
    body = {
        "model": model_name,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": 0.95,
        "thinking_budget": thinking_budget,
    }
    if enable_thinking:
        body["enable_thinking"] = True

    return {
        "custom_id": custom_id,
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": body
    }

def gen_step1(args):
    print("Generating Stage 1 batch requests...")
    swe_bench_data = load_dataset(args.dataset, split="test")
    
    with open(args.output_file, 'w') as f:
        for instance_data in tqdm(swe_bench_data):
            instance_id = instance_data['instance_id']
            problem_statement = instance_data["problem_statement"]
            if args.convert_crlf:
                problem_statement = problem_statement.replace('\r\n', '\n')

            structure = get_repo_structure(instance_id, args.repostructure_dir)
            
            if structure is None:
                continue

            obtain_relevant_files_prompt = """
    Please look through the following GitHub problem description and Repository structure and provide a list of files that one would need to edit to fix the problem.

    ### GitHub Problem Description ###
    {problem_statement}

    ###

    ### Repository Structure ###
    {structure}

    ###

    Please only provide the full path and return at most 5 files.
    The returned files should be separated by new lines ordered by most to least important and wrapped with ```
    For example:
    ```
    file1.py
    file2.py
    ```
    """
            prompt_content = obtain_relevant_files_prompt.format(
                problem_statement=problem_statement,
                structure=show_project_structure(structure).strip(),
            ).strip()
            
            messages = [{"role": "user", "content": prompt_content}]
            
            for i in range(args.passk):
                # If passk > 1, append suffix. If passk=1, we can optionally append or not. 
                # User asked to "Keep track of the instance within the process using custom_id".
                # For consistency let's append suffix if passk > 1 or just always append?
                # Usually if passk=1 we might want clean IDs inside, but for uniform processing later:
                # Let's use suffix always if args.passk > 1.
                # Or simplier: always use suffix if we want to be consistent, but that changes output format for single pass.
                # Let's use suffix always for simplicity in logic, or check args.passk.
                
                cid = instance_id
                if args.passk > 1:
                     cid = f"{instance_id}__pass{i}"
                
                request = make_request_body(
                    custom_id=cid, 
                    model_name=args.model_name, 
                    messages=messages, 
                    max_tokens=args.max_tokens, 
                    temperature=args.temp,
                    enable_thinking=args.enable_thinking,
                    thinking_budget=args.thinking_budget
                )
                f.write(json.dumps(request) + '\n')
    print(f"Stage 1 requests saved to {args.output_file}")



def process_item_step2(item_tuple):
    custom_id, response_item, instance_data, args_repostructure_dir, args_convert_crlf, args_model_name, args_max_tokens, args_temp, args_enable_thinking, args_thinking_budget = item_tuple
    
    if 'status_code' in response_item['response'] and response_item['response']['status_code'] != 200:
        # print(f"Error in response for {custom_id}: {response_item['response']}")
        return None

    raw_answer = response_item['response']['body']['choices'][0]['message']['content']
    
    # We need instance_id to get structure
    # instance_data has the ID.
    instance_id = instance_data['instance_id']

    structure = get_repo_structure(instance_id, args_repostructure_dir)
    if structure is None: 
        return None

    # Post-processing to find files
    model_found_files = raw_answer.strip().split("\n")
    files, classes, functions = get_full_file_paths_and_classes_and_functions(structure)
    model_found_files = [correct_file_path_in_structure(file, structure) for file in model_found_files]
    found_files = correct_file_paths(model_found_files, files)
   
    # Prepare Stage 2 Prompt
    repair_relevant_file_instruction = """
    Below are some code segments, each from a relevant file. One or more of these files may contain bugs.
    """
    repair_prompt_combine_topn_cot_diff = """
    We are currently solving the following issue within our repository. Here is the issue text:
    --- BEGIN ISSUE ---
    {problem_statement}
    --- END ISSUE ---

    {repair_relevant_file_instruction}
    --- BEGIN FILE ---
    ```
    {content}
    ```
    --- END FILE ---

    Please first localize the bug based on the issue statement, and then generate *SEARCH/REPLACE* edits to fix the issue.

    Every *SEARCH/REPLACE* edit must use this format:
    1. The file path
    2. The start of search block: <<<<<<< SEARCH
    3. A contiguous chunk of lines to search for in the existing source code
    4. The dividing line: =======
    5. The lines to replace into the source code
    6. The end of the replace block: >>>>>>> REPLACE

    Here is an example:

    ```python
    ### mathweb/flask/app.py
    <<<<<<< SEARCH
    from flask import Flask
    =======
    import math
    from flask import Flask
    >>>>>>> REPLACE
    ```

    Please note that the *SEARCH/REPLACE* edit REQUIRES PROPER INDENTATION. If you would like to add the line '        print(x)', you must fully write that out, with all those spaces before the code!
    Wrap the *SEARCH/REPLACE* edit in blocks ```python...```.
    """
    problem_statement = instance_data["problem_statement"]
    if args_convert_crlf:
        problem_statement = problem_statement.replace('\r\n', '\n')

    file_contents = get_repo_files(structure, found_files)
    contents = ""
    for file_name, content in file_contents.items():
        contents += f"{file_name}\n{content}\n\n"

    prompt_content = repair_prompt_combine_topn_cot_diff.format(
        repair_relevant_file_instruction=repair_relevant_file_instruction,
        problem_statement=problem_statement,
        content=contents.rstrip(),
    ).strip()

    
    # Single-turn request
    messages = [
        {"role": "user", "content": prompt_content}
    ]
    
    request = make_request_body(
        custom_id=custom_id, # Preserve custom_id with pass suffix
        model_name=args_model_name,
        messages=messages,
        max_tokens=args_max_tokens,
        temperature=args_temp,
        enable_thinking=args_enable_thinking,
        thinking_budget=args_thinking_budget
    )
    return json.dumps(request)


def proc_step1_gen_step2(args):
    print("Processing Stage 1 outputs and generating Stage 2 requests...")
    
    # Load batch outputs
    batch_outputs = {}
    with open(args.input_file, 'r') as f:
        for line in f:
            item = json.loads(line)
            batch_outputs[item['custom_id']] = item

    swe_bench_data = load_dataset(args.dataset, split="test")
    # Optimize lookups
    swe_bench_dict = {d['instance_id']: d for d in swe_bench_data}
    
    process_items = []
    
    # Prepare items for parallel processing
    for custom_id, response_item in batch_outputs.items():
         # Parse instance_id
         if "__pass" in custom_id:
             instance_id = custom_id.split("__pass")[0]
         else:
             instance_id = custom_id
         
         if instance_id not in swe_bench_dict:
             continue
         
         instance_data = swe_bench_dict[instance_id]
         
         process_items.append((
             custom_id, 
             response_item, 
             instance_data, 
             args.repostructure_dir, 
             args.convert_crlf, 
             args.model_name, 
             args.max_tokens, 
             args.temp, 
             args.enable_thinking, 
             args.thinking_budget
         ))

    print(f"Prepared {len(process_items)} items for processing. Starting parallel execution...")

    with open(args.output_file, 'w') as f:
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.num_workers) as executor:
            # Use chunks to improve performance if many small tasks, though here tasks are heavy-ish.
            # Default chunksize is usually fine, but explicit can be better.
            results = list(tqdm(executor.map(process_item_step2, process_items), total=len(process_items)))
            
            for res in results:
                if res:
                    f.write(res + '\n')
            
    print(f"Stage 2 requests saved to {args.output_file}")


def proc_step2(args):
    print("Processing Stage 2 outputs and saving final results...")
    
    batch_outputs = {}
    with open(args.input_file, 'r') as f:
        for line in f:
            item = json.loads(line)
            batch_outputs[item['custom_id']] = item

    swe_bench_data = load_dataset(args.dataset, split="test")
    swe_bench_dict = {d['instance_id']: d for d in swe_bench_data}
    
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
        

def process_item_step3(item_tuple):
    custom_id, response_item, instance_data, args_repostructure_dir, args_save_dir = item_tuple

    if 'status_code' in response_item['response'] and response_item['response']['status_code'] != 200:
            # print(f"Error in response for {custom_id}: {response_item['response']}")
            return None

    raw_answer = response_item['response']['body']['choices'][0]['message']['content']
    search_replace_text = raw_answer 
    
    def post_process_response(response: str) -> str:
        content = response
        if "◁/think▷" in content:
            content = content.replace("◁think▷", "")
            parts = content.split("◁/think▷")
            content = parts[-1]
        elif "</think>" in content:
            parts = content.split("</think>")
            content = parts[-1]
        
        import re
        matches = re.findall(r"```.*?```", content, re.DOTALL)
        if matches:
            return "\n".join(matches)
        return content

    search_replace_text = post_process_response(raw_answer)
    
    instance_id = instance_data['instance_id']
    structure = get_repo_structure(instance_id, args_repostructure_dir)
    if structure is None: return None

    # Generate Patch
    model_patch, edited_files, original_contents, new_contents = generate_robust_diff(structure=structure, search_replace_text=search_replace_text)
    
    # Prepare output dict
    gt_patch = instance_data.get("patch")
    from kimidev.agentlessnano.utils import parse_patch, get_relevant_files, generate_found_edit_locs
    gt_parsed_patch = parse_patch(gt_patch)
    instance_data["parsed_patch"] = gt_parsed_patch
    found_files = get_relevant_files(instance_data)
    found_related_locs, found_edit_locs = generate_found_edit_locs(gt_parsed_patch, structure)
    
    gt_results = {
        "instance_id": instance_id,
        "found_files": found_files,
        "found_related_locs": found_related_locs,
        "found_edit_locs": found_edit_locs,
        "gt_parsed_patch": gt_parsed_patch,
        "gt_patch": gt_patch,
    }

    # Save result
    # Save result
    # 1. Raw/Ungrouped dict (previous format, for debugging/reference)
    raw_dict = {
        "instance_id": instance_id,
        "model_patch": model_patch,
        "search_replace_text": search_replace_text,
        "raw_output": raw_answer, # Save raw output for debugging
        "gt_results": gt_results
    }
    
    # 2. Processed dict (Agentless repair schema)
    processed_dict = {
        "model_name_or_path": "agentless",
        "instance_id": instance_id,
        "model_patch": model_patch,
        "raw_model_patch": search_replace_text,
        "original_file_content": original_contents,
        "edited_files": edited_files,
        "new_file_content": new_contents
    }
    
    return (custom_id, raw_dict, processed_dict)


def proc_step2(args):
    print("Processing Stage 2 outputs and saving final results...")
    
    batch_outputs = {}
    with open(args.input_file, 'r') as f:
        for line in f:
            item = json.loads(line)
            batch_outputs[item['custom_id']] = item

    swe_bench_data = load_dataset(args.dataset, split="test")
    swe_bench_dict = {d['instance_id']: d for d in swe_bench_data}
    
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    
    process_items = []
    
    for custom_id, response_item in batch_outputs.items():
        if "__pass" in custom_id:
             instance_id = custom_id.split("__pass")[0]
        else:
             instance_id = custom_id
             
        if instance_id not in swe_bench_dict:
            continue

        instance_data = swe_bench_dict[instance_id]
        
        process_items.append((
            custom_id,
            response_item,
            instance_data,
            args.repostructure_dir,
            args.save_dir
        ))
        
    print(f"Prepared {len(process_items)} items for processing. Starting parallel execution...")

    with concurrent.futures.ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        results = list(tqdm(executor.map(process_item_step3, process_items), total=len(process_items)))
        
        
        # Aggregate results by pass index
        pass_outputs = {} # integer key -> list of json strings
        
        # Subdirectory for ungrouped samples
        samples_dir = os.path.join(args.save_dir, 'samples')
        if not os.path.exists(samples_dir):
            os.makedirs(samples_dir)
            
        for res in results:
            if res:
                custom_id, raw_dict, processed_dict = res
                
                # 1. Save ungrouped raw version to subdirectory
                output_file_path = os.path.join(samples_dir, f'{custom_id}.jsonl')
                with open(output_file_path, 'w') as out_f:
                    out_f.write(json.dumps(raw_dict, ensure_ascii=False) + '\n')
                
                # 2. Aggregate processed version
                # Determine pass index
                if "__pass" in custom_id:
                    try:
                        pass_idx = int(custom_id.split("__pass")[1])
                    except ValueError:
                        pass_idx = 0
                else:
                    pass_idx = 0
                
                if pass_idx not in pass_outputs:
                    pass_outputs[pass_idx] = []
                
                pass_outputs[pass_idx].append(json.dumps(processed_dict))
            
        # Save aggregated files
        for pass_idx, lines in pass_outputs.items():
            output_filename = f"output_{pass_idx}_processed.jsonl"
            output_file_path = os.path.join(args.save_dir, output_filename)
            
            with open(output_file_path, 'w') as out_f:
                for line in lines:
                    out_f.write(line + '\n')
            
            print(f"Saved {len(lines)} records to {output_filename}")
            
    print(f"All results saved to {args.save_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='stage', required=True)
    
    # Stage 1
    parser_s1 = subparsers.add_parser('gen_step1')
    parser_s1.add_argument('--dataset', default='princeton-nlp/SWE-bench_Verified')
    parser_s1.add_argument('--repostructure_dir', required=True)
    parser_s1.add_argument('--output_file', required=True)
    parser_s1.add_argument('--model_name', default='deepseek-ai/DeepSeek-R1')
    parser_s1.add_argument('--max_tokens', type=int, default=32768)
    parser_s1.add_argument('--temp', type=float, default=1.0)
    parser_s1.add_argument('--enable_thinking', action='store_true', help='Enable thinking')
    parser_s1.add_argument('--thinking_budget', type=int, default=32768, help='Thinking budget')
    parser_s1.add_argument('--convert_crlf', action='store_true', help='Convert CRLF to LF in problem statement')
    parser_s1.add_argument('--passk', type=int, default=1, help='Number of samples to generate per instance')
    
    # Stage 2
    parser_s2 = subparsers.add_parser('proc_step1_gen_step2')
    parser_s2.add_argument('--dataset', default='princeton-nlp/SWE-bench_Verified')
    parser_s2.add_argument('--repostructure_dir', required=True)
    parser_s2.add_argument('--input_file', required=True, help="Output of Stage 1 batch")
    parser_s2.add_argument('--output_file', required=True, help="Input for Stage 2 batch")
    parser_s2.add_argument('--model_name', default='deepseek-ai/DeepSeek-R1')
    parser_s2.add_argument('--max_tokens', type=int, default=32768)
    parser_s2.add_argument('--temp', type=float, default=1.0)
    parser_s2.add_argument('--enable_thinking', action='store_true', help='Enable thinking')
    parser_s2.add_argument('--thinking_budget', type=int, default=32768, help='Thinking budget')
    parser_s2.add_argument('--convert_crlf', action='store_true', help='Convert CRLF to LF in problem statement')
    parser_s2.add_argument('--num_workers', type=int, default=8, help='Number of worker processes for parallel processing')
    
    # Stage 3
    parser_s3 = subparsers.add_parser('proc_step2')
    parser_s3.add_argument('--dataset', default='princeton-nlp/SWE-bench_Verified')
    parser_s3.add_argument('--repostructure_dir', required=True)
    parser_s3.add_argument('--input_file', required=True, help="Output of Stage 2 batch")
    parser_s3.add_argument('--save_dir', required=True)
    parser_s3.add_argument('--num_workers', type=int, default=8, help='Number of worker processes for parallel processing')

    args = parser.parse_args()
    
    if args.stage == 'gen_step1':
        gen_step1(args)
    elif args.stage == 'proc_step1_gen_step2':
        proc_step1_gen_step2(args)
    elif args.stage == 'proc_step2':
        proc_step2(args)
