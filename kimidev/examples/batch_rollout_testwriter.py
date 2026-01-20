
import json
import os
import argparse
from tqdm import tqdm
import sys
import functools
import concurrent.futures

from datasets import load_dataset
from kimidev.agentlessnano.utils import (
    search_instance_id_and_extract_structure,
    get_repo_files,
    correct_file_path_in_structure,
    correct_file_paths,
    get_full_file_paths_and_classes_and_functions,
    show_project_structure
)
from kimidev.agentlessnano.testwritter_utils import remove_test_cases, create_patch_from_code
from kimidev.agentlessnano.post_process import generate_model_patch_difflib_testwritter

@functools.lru_cache(maxsize=50)
def get_repo_structure(instance_id, repostructure_dir):
    return search_instance_id_and_extract_structure(instance_id, repostructure_dir)

def make_request_body(custom_id, model_name, messages, max_tokens, temperature, enable_thinking=False, thinking_budget=16000):
    body = {
        "model": model_name,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if enable_thinking:
        body["enable_thinking"] = True
        body["thinking_budget"] = thinking_budget
        
    request = {
        "custom_id": custom_id,
        "method": "POST",
        "url": "/chat/completions",
        "body": body
    }
    return request

def gen_step1(args):
    print("Generating Stage 1 requests (Relevant Test Files)...")
    
    swe_bench_data = load_dataset(args.dataset, split="test")
    
    with open(args.output_file, 'w') as f:
        for instance_data in tqdm(swe_bench_data):
            instance_id = instance_data["instance_id"]
            problem_statement = instance_data["problem_statement"]
            
            if args.convert_crlf:
                problem_statement = problem_statement.replace('\r\n', '\n')
            
            structure = get_repo_structure(instance_id, args.repostructure_dir)
            if structure is None:
                print(f"Skipping {instance_id}: Structure not found")
                continue

            obtain_relevant_files_prompt = """
Please look through the following GitHub problem description and Repository structure and provide a list of test files that should be run after applying the patch to fix the issue.

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
            # Note: test_flag=True for testwriter
            prompt_content = obtain_relevant_files_prompt.format(
                problem_statement=problem_statement,
                structure=show_project_structure(structure, test_flag=True).strip(),
            ).strip()
            
            messages = [{"role": "user", "content": prompt_content}]
            
            for i in range(args.passk):
                custom_id = f"{instance_id}__pass{i}" if args.passk > 1 else instance_id
                
                # Temperature handling: 0.0 for first pass, 1.0 for others (if passk > 1 logic from original script)
                # But here we stick to args.temp or specific logic? 
                # Original rollout_messages_testwriter.py: args.temp = 0.0 if pass_idx == 0 else 1.0
                current_temp = 0.0 if i == 0 else 1.0
                # If user supplied specific temp via args, maybe we should respect it? 
                # The original script overrides it. Let's follow original script logic for passk > 1
                if args.passk > 1:
                     temp = current_temp
                else:
                     temp = args.temp

                request = make_request_body(
                    custom_id=custom_id, 
                    model_name=args.model_name, 
                    messages=messages,
                    max_tokens=args.max_tokens,
                    temperature=temp, 
                    enable_thinking=args.enable_thinking,
                    thinking_budget=args.thinking_budget
                )
                f.write(json.dumps(request) + '\n')
                
    print(f"Stage 1 requests saved to {args.output_file}")


def process_item_step2(item_tuple):
    custom_id, response_item, instance_data, args_repostructure_dir, args_convert_crlf, args_model_name, args_max_tokens, args_temp, args_enable_thinking, args_thinking_budget = item_tuple
    
    # Optional status code check
    if 'status_code' in response_item['response'] and response_item['response']['status_code'] != 200:
        # print(f"Error in response for {custom_id}: {response_item['response']}")
        return None

    raw_answer = response_item['response']['body']['choices'][0]['message']['content']
    
    if "__pass" in custom_id:
        instance_id = custom_id.split("__pass")[0]
        pass_idx = int(custom_id.split("__pass")[1])
    else:
        instance_id = custom_id
        pass_idx = 0

    structure = get_repo_structure(instance_id, args_repostructure_dir)
    if structure is None: return None

    # Post-processing to find files
    model_found_files = raw_answer.strip().split("\n")
    files, classes, functions = get_full_file_paths_and_classes_and_functions(structure)
    model_found_files = [correct_file_path_in_structure(file, structure) for file in model_found_files]
    found_files = correct_file_paths(model_found_files, files)
    
    # Filter for test files only
    pred_found_files_new = []
    for file in found_files:
        if 'test' in file:
            pred_found_files_new.append(file)
    found_files = pred_found_files_new

    # Take top 1
    if len(found_files) > 0:
        found_files = found_files[:1]
    else:
        # If no test files found, maybe return valid request with empty file list? 
        # But repair needs contents. Original script: repair_message = [], search_replace_text = ""
        # If we skip here, we produce no request for this item.
        # But we must align with original logic: if empty, it skips LLM call and returns empty result.
        # In batch, we can't skip LLM call mid-way easily if we want to produce a file. 
        # Actually, if we produce nothing here, step 3 won't have input. 
        # So we probably should skip generating a request.
        return None

    # Prepare Stage 2 Prompt
    repair_relevant_file_instruction = """
Below are some code segments, each from a relevant test file. One or more of these files may be added some new tests which can reproduce the issue .
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

Please first localize some possible locations in those test files within the repo, and then generate *SEARCH/REPLACE* edit updates to the **test** files in the repo, so that the erroneous scenario described in the problem is reproduced.

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

def test__rules__std_L060_raised() -> None:
    try:
        sql = "SELECT   IFNULL(NULL, 100),
            NVL(NULL,100);"
        result = lint(sql, rules=["L060"])
        assert len(result) == 2
    except:
        print("Other issues")
        return

    try:
        assert result[0]["description"] == "Use 'COALESCE' instead of 'IFNULL'."
        assert result[1]["description"] == "Use 'COALESCE' instead of 'NVL'."
        print("Issue resolved")
    except AssertionError:
        print("Issue reproduced")
        return

    return
>>>>>>> REPLACE
```

Please note that the *SEARCH/REPLACE* edit REQUIRES PROPER INDENTATION. If you would like to add the line '        print(x)', you must fully write that out, with all those spaces before the code!
Wrap the *SEARCH/REPLACE* edit in blocks ```python...```.
"""
    problem_statement = instance_data["problem_statement"]
    if args_convert_crlf:
        problem_statement = problem_statement.replace('\r\n', '\n')

    file_contents = get_repo_files(structure, found_files)
    file_contentes_dict = remove_test_cases(structure, {}) # SWE_BENCH_VERIFIED does not need to remove test cases? Original script calls it.
    
    contents = ""
    for file_name, content in file_contents.items():
        if(file_name in file_contentes_dict):
            content = file_contentes_dict[file_name]['new_content']
        else:
            file_contentes_dict[file_name] = {
                'old_content': content,
                'new_content': content
            }
        contents += f"{file_name}\n{content}\n\n"

    prompt_content = repair_prompt_combine_topn_cot_diff.format(
        repair_relevant_file_instruction=repair_relevant_file_instruction,
        problem_statement=problem_statement,
        content=contents.rstrip(),
    ).strip()

    
    messages = [
        {"role": "user", "content": prompt_content}
    ]
    
    # Temp logic similar to gen_step1 (0.0 for pass0, 1.0 for others)
    current_temp = 0.0 if pass_idx == 0 else 1.0
    # Use args_temp effectively? Or stick to original logic? 
    # Original script uses specific temp logic for passes.
    # We will assume args_temp is passed but maybe we should override if pass_idx > 0?
    if "__pass" in custom_id: # Implies passk > 1 logic invoked 
         temp = current_temp
    else:
         temp = args_temp

    request = make_request_body(
        custom_id=custom_id, # Preserve custom_id with pass suffix
        model_name=args_model_name,
        messages=messages,
        max_tokens=args_max_tokens,
        temperature=temp,
        enable_thinking=args_enable_thinking,
        thinking_budget=args_thinking_budget
    )
    return json.dumps(request)


def proc_step1_gen_step2(args):
    print("Processing Stage 1 outputs and generating Stage 2 requests...")
    
    batch_outputs = {}
    with open(args.input_file, 'r') as f:
        for line in f:
            item = json.loads(line)
            batch_outputs[item['custom_id']] = item

    swe_bench_data = load_dataset(args.dataset, split="test")
    swe_bench_dict = {d['instance_id']: d for d in swe_bench_data}
    
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
            results = list(tqdm(executor.map(process_item_step2, process_items), total=len(process_items)))
            
            for res in results:
                if res:
                    f.write(res + '\n')
            
    print(f"Stage 2 requests saved to {args.output_file}")


def process_item_step3(item_tuple):
    custom_id, response_item, instance_data, args_repostructure_dir, args_save_dir = item_tuple

    # Optional status code check
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
    
    if "__pass" in custom_id:
        instance_id = custom_id.split("__pass")[0]
    else:
        instance_id = custom_id

    structure = get_repo_structure(instance_id, args_repostructure_dir)
    if structure is None: return None

    # RECONSTRUCT CONTEXT for generate_model_patch_difflib_testwritter
    # We need file_contentes_dict used in step 2. 
    # We must reproduce the logic to find files.
    
    # Step 1 response reconstruction NOT NEEDED, we just need to re-find the file to build context.
    # WAIT - step 2 response is just the patch. To run `generate_model_patch_difflib_testwritter`,
    # we need `file_contentes_dict`.
    # `file_contentes_dict` comes from `found_files` which comes from Step 1 response.
    # But we don't have Step 1 response here easily unless we pass it through or re-read Step 1 batch output.
    # Or strict adherence: The model output `search_replace_text` contains the file path usually...
    # `generate_model_patch_difflib_testwritter` arguments: `file_contentes_dict`, `search_replace_text`.
    # `file_contentes_dict` maps filename -> {old_content, new_content}.
    
    # Problem: In `proc_step2`, we only have Step 2 output. We don't know what file was selected in Step 1 
    # unless we parse it from the prompt (request) or we rely on `search_replace_text` containing the file path.
    # The `search_replace_text` (LLM output) *should* contain the file path as per instructions.
    # But `generate_model_patch_difflib_testwritter` needs the original content to verify/patch?
    # Let's check `generate_model_patch_difflib_testwritter` implementation if I can... 
    # Actually I can't check it easily right now without breaking flow. 
    # But `rollout_messages_testwriter.py` generates `file_contentes_dict` BEFORE calling LLM.
    
    # Ideally `proc_step2` needs to reproduce the `found_files` logic. 
    # But `found_files` came from Step 1 response.
    # We don't have Step 1 response in `proc_step2` input (which is Step 2 response).
    # Step 2 REQUEST contained the context. Step 2 RESPONSE contains the patch.
    # We need to map CustomID -> Step 1 Response to reproduce the file selection?
    # Or we can assume the file path is in the patch?
    
    # In `batch_rollout_bugfixer.py`, `proc_step2` (step 3 here) calls `generate_model_patch_difflib`.
    # That function usually takes structure and text.
    # Here `generate_model_patch_difflib_testwritter` takes `file_contentes_dict`.
    # We NEED to reconstruct `file_contentes_dict`. 
    # This implies we need the `found_files`.
    # `found_files` comes from Step 1 output.
    
    # SOLUTION: In `proc_step2`, we need to read not just Step 2 output, but also Step 1 output (or pass it along?).
    # Standard batch flow involves independent stages. 
    # BUT, we can just require `step1_output_file` as an argument to `proc_step2`? 
    # OR, better: We can parse the `custom_id` -> find `instance_id`.
    # But we don't know which file was selected by Model 1.
    # Wait, in the original script `rollout_messages_testwriter.py`:
    # `pred_found_files` comes from `relevant_file_prompt_response`.
    # Then `file_contentes_dict` is built.
    # Then Model 2 is called.
    # Then `generate_model_patch_difflib_testwritter` is called.
    
    # So `proc_step2` needs the result of Model 1.
    # We can ask user to provide Step 1 output file as well? 
    # `batch_rollout_bugfixer.py` `proc_step2` calls `generate_model_patch_difflib(structure, search_replace_text)`. 
    # It parses file path FROM the `search_replace_text` (LLM output).
    # Does `generate_model_patch_difflib_testwritter` support that?
    # Inspecting `rollout_messages_testwriter.py` line 296:
    # `model_patch = generate_model_patch_difflib_testwritter(file_contentes_dict=file_contentes_dict, search_replace_text=search_replace_text)`
    # It requires `file_contentes_dict`.
    
    # We need to reconstruct `file_contentes_dict`.
    # To do that, we need `found_files`.
    # `found_files` used in Stage 2 generation was:
    # 1. Parse Step 1 response. 2. Filter test files. 3. Top 1.
    
    # So `proc_step2` MUST have access to Step 1 response.
    # I will add `--step1_file` argument to `proc_step2`.
    # And we will load it to map `custom_id` -> `step1_response`.
    pass

    # Placeholder for logic implemented below in the full text string
    # ...

def proc_step2(args):
    print("Processing Stage 2 outputs and saving final results...")
    
    batch_outputs = {}
    with open(args.input_file, 'r') as f:
        for line in f:
            item = json.loads(line)
            batch_outputs[item['custom_id']] = item

    # Load Step 1 outputs to reconstruct context
    step1_outputs = {}
    if args.step1_file:
        with open(args.step1_file, 'r') as f:
            for line in f:
                item = json.loads(line)
                step1_outputs[item['custom_id']] = item
    else:
        print("Error: --step1_file is required for Step 3 to reconstruct context (found_files).")
        return

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
            
        if custom_id not in step1_outputs:
            print(f"Warning: Step 1 output not found for {custom_id}")
            continue

        instance_data = swe_bench_dict[instance_id]
        step1_item = step1_outputs[custom_id]
        
        process_items.append((
            custom_id,
            response_item,
            step1_item,
            instance_data,
            args.repostructure_dir,
            args.save_dir
        ))
        
    print(f"Prepared {len(process_items)} items for processing. Starting parallel execution...")

    # Subdirectory for ungrouped samples
    samples_dir = os.path.join(args.save_dir, 'samples')
    os.makedirs(samples_dir, exist_ok=True)
    
    pass_outputs = {} # integer key -> list of json strings

    with concurrent.futures.ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        # Submit all tasks
        futures = {executor.submit(process_item_step3_full, item): item[0] for item in process_items}
        
        # Failure log file
        failure_log_path = os.path.join(args.save_dir, 'failures.jsonl')
        
        # Pass failure_log_path into worker (or write inside worker locally? 
        # Writing to single file from multiple workers is tricky (race conditions/interleaved writes).
        # Better to have each worker write its own failures or just return failure info (small).
        # But saving SUCCESS files (which are unique) is safe in worker.
        
        # Updated plan: Save SUCCESS files in worker. Return FAILURE info to main (it's small).
        
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(process_items)):
            try:
                res = future.result()
                if res:
                    if res.get('status') == 'failed':
                         # Log failure in MAIN process to avoid race conditions on single file
                        with open(failure_log_path, 'a') as f:
                            f.write(json.dumps(res, ensure_ascii=False) + '\n')
                        continue
                    
                    if res.get('status') == 'success':
                         # Files already saved in worker
                         # Aggregate for final output if needed? 
                         # We need to build pass_outputs for "output_N_processed.jsonl"
                         # So we still need minimal info: custom_id (for pass_idx) and instance_dict (parsed content).
                         # But instance_dict has patch which can be large.
                         # Can we write to "output_N" incrementally? No, it's one file per pass.
                         # Maybe we just need the JSON line string?
                         
                         custom_id = res['custom_id']
                         instance_dict_str = res['instance_dict_str'] # Pass string instead of dict?
                         
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
                         
                         pass_outputs[pass_idx].append(instance_dict_str)

            except Exception as e:
                print(f"Error processing item: {e}")
            except Exception as e:
                print(f"Error processing item: {e}")
        
    # Save aggregated files
    for pass_idx, lines in pass_outputs.items():
        output_filename = f"output_{pass_idx}_processed_reproduction_test.jsonl"
        output_file_path = os.path.join(args.save_dir, output_filename)
        
        with open(output_file_path, 'w') as out_f:
            for line in lines:
                out_f.write(line + '\n')
        
        print(f"Saved {len(lines)} records to {output_filename}")
            
    print(f"All results saved to {args.save_dir}")


def fix_relative_imports(content, file_path):
    """
    Converts relative imports (e.g. 'from . import utls') to absolute imports 
    (e.g. 'from path.to import utils') based on the file_path.
    """
    import ast
    
    # Estimate package path from file_path
    # e.g., 'django/contrib/admin/tests.py' -> 'django.contrib.admin'
    # Remove extension
    path_no_ext = os.path.splitext(file_path)[0]
    parts = path_no_ext.split(os.sep)
    
    # If the file is basically at root, no relative imports to fix (or stripped)
    if len(parts) <= 1:
        return content

    # The package this file belongs to is the dir path
    package_parts = parts[:-1]
    package_name = ".".join(package_parts)
    
    try:
        tree = ast.parse(content)
        
        class ImportFixer(ast.NodeTransformer):
            def visit_ImportFrom(self, node):
                if node.level > 0: # Relative import
                    # level 1 = ., level 2 = ..
                    # We need to go up (level - 1) times from package_name
                    
                    # Effective package parts
                    if node.level > len(package_parts) + 1:
                         # Too many dots? Just return (can't resolve)
                         return node
                    
                    # . (level 1) means current package
                    # .. (level 2) means parent
                    # so we slice package_parts[:-(level-1)]
                    
                    if node.level == 1:
                        base_pkg = package_parts
                    else:
                        base_pkg = package_parts[:-(node.level - 1)]
                        
                    base_pkg_str = ".".join(base_pkg)
                    
                    if node.module:
                        new_module = f"{base_pkg_str}.{node.module}"
                    else:
                        new_module = base_pkg_str
                        
                    node.level = 0
                    node.module = new_module
                return node
        
        fixed_tree = ImportFixer().visit(tree)
        ast.fix_missing_locations(fixed_tree)
        return ast.unparse(fixed_tree)
        
    except Exception as e:
        print(f"Error fixing relative imports: {e}")
        return content

def make_reproduction_script(content, test_func_name, file_path=None):
    """
    Wraps the content (which contains the new test) into a standalone script
    that runs the specific test function using pytest or unittest.
    Also fixes relative imports if file_path is provided.
    """
    import ast
    
    # Fix relative imports first
    if file_path:
        content = fix_relative_imports(content, file_path)
    
    # Check if we can find the class of the function
    try:
        tree = ast.parse(content)
        class_name = None
        
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                for subnode in node.body:
                    if isinstance(subnode, ast.FunctionDef) and subnode.name == test_func_name:
                        class_name = node.name
                        break
                if class_name:
                    break
        
        script_content = content + "\n\n"
        script_content += "if __name__ == '__main__':\n"
        
        if class_name:
            # It's a method in a class. Try to distinguish unittest vs pytest style?
            # Safe bet: use unittest runner if it looks like unittest, otherwise try pytest
            # But simpler: just use unittest runner with correct address if possible, 
            # Or assume pytest is available in environment (standard for swe-bench verified?)
            
            # Using unittest invocation style which handles both usually if it inherits TestCase
            script_content += "    import unittest\n"
            script_content += "    try:\n"
            script_content += f"        suite = unittest.TestLoader().loadTestsFromName('__main__.{class_name}.{test_func_name}')\n"
            script_content += "        res = unittest.TextTestRunner(verbosity=2).run(suite)\n"
            script_content += "        if not res.wasSuccessful():\n"
            script_content += "             print('Issue reproduced')\n"
            script_content += "             sys.exit(1)\n"
            script_content += "        else:\n"
            script_content += "             print('Issue resolved')\n"
            script_content += "             sys.exit(0)\n"
            script_content += "    except ImportError:\n"
            # Fallback to pytest if unittest loading fails (e.g. not a TestCase)
            script_content += "        import pytest\n"
            script_content += "        import sys\n"
            script_content += f"        ret = pytest.main(['-v', __file__ + '::{class_name}::{test_func_name}'])\n"
            script_content += "        if ret != 0:\n"
            script_content += "            print('Issue reproduced')\n"
            script_content += "        else:\n"
            script_content += "            print('Issue resolved')\n"
            script_content += "        sys.exit(ret)\n"
            script_content += "    except Exception:\n"
             # Double fallback
            script_content += "        import pytest\n"
            script_content += "        import sys\n"
            script_content += f"        ret = pytest.main(['-v', __file__ + '::{class_name}::{test_func_name}'])\n"
            script_content += "        if ret != 0:\n"
            script_content += "            print('Issue reproduced')\n"
            script_content += "        else:\n"
            script_content += "            print('Issue resolved')\n"
            script_content += "        sys.exit(ret)\n"
        else:
            # Top level function
            script_content += f"    try:\n"
            script_content += f"        {test_func_name}()\n"
            script_content += "        print('Issue resolved')\n"
            script_content += "        sys.exit(0)\n"
            script_content += "    except Exception:\n"
            script_content += "        print('Issue reproduced')\n"
            script_content += "        sys.exit(1)\n"
            
        return script_content
    except Exception as e:
        # Fallback: just return content and hope
        print(f"Error making reproduction script: {e}")
        return content

def apply_patches(file_contentes_dict, search_replace_text):
    """
    Parses SEARCH/REPLACE blocks and updates file_contentes_dict in place.
    """
    import re
    # Pattern to match:
    # [filename]
    # <<<<<<< SEARCH
    # [search_content]
    # =======
    # [replace_content]
    # >>>>>>> REPLACE
    
    # Using the pattern from post_process.py
    pattern = r'(?:### )?([^\n]+)\n<{4,} SEARCH\n(.*?)\n={4,}\n(.*?)\n>{4,} REPLACE'
    
    matches = re.findall(pattern, search_replace_text, re.DOTALL)
    
    for file_name, search_str, replace_str in matches:
        file_name = file_name.strip()
        if file_name in file_contentes_dict:
            current_content = file_contentes_dict[file_name]['new_content']
            
            # 1. Exact match
            if search_str in current_content:
                new_content = current_content.replace(search_str, replace_str)
                file_contentes_dict[file_name]['new_content'] = new_content
                continue
                
            # 2. Try normalizing match (strip start/end newlines)
            s_strip = search_str.strip()
            if s_strip and s_strip in current_content:
                # If the stripped search block is unique enough to be found:
                # match replace_str behavior? 
                # Attempt to replace the literal found stripped string with the replace string (maybe also stripped?)
                # This helps if the model added/missed a newline at start/end of block.
                # However, we must be careful not to leave artifacts.
                
                # Heuristic: If replace_str starts/ends with newline matching search_str, preserve?
                # Simpler: just replace the found s_strip with replace_str.strip() surrounded by necessary newlines?
                # Let's just try replacing s_strip with replace_str (preserving its indentation if it had any).
                
                # Check if replacing s_strip gives a valid python code approx?
                # Actually, often the model output has `\n\n` before search, regex captures empty line.
                # s_strip removes it. current_content has the code.
                # So replacing s_strip with replace_str is plausible.
                new_content = current_content.replace(s_strip, replace_str)
                file_contentes_dict[file_name]['new_content'] = new_content
                continue
            
            print(f"Warning: SEARCH block not found in {file_name}")

def make_reproduction_script_full_file(content, file_path=None):
    """
    Uses the full file content as the reproduction script.
    Fixes relative imports and ensures a main block exists to run tests.
    """
    if file_path:
        content = fix_relative_imports(content, file_path)
        
    script_content = content + "\n\n"
    
    # Check if main block already exists
    if "if __name__ == '__main__':" not in content and 'if __name__ == "__main__":' not in content:
        script_content += "if __name__ == '__main__':\n"
        script_content += "    try:\n"
        script_content += "        import pytest\n"
        script_content += "        import sys\n"
        script_content += "        ret = pytest.main(['-v', __file__])\n"
        script_content += "        if ret != 0:\n"
        script_content += "            print('Issue reproduced')\n"
        script_content += "        else:\n"
        script_content += "            print('Issue resolved')\n"
        script_content += "        sys.exit(ret)\n"
        script_content += "    except ImportError:\n"
        script_content += "        import unittest\n"
        # unittest.main typically exits. We use exit=False to capture result
        script_content += "        try:\n"
        script_content += "            unittest.main(exit=False)\n"
        script_content += "            # If we get here without exception, tests passed? No, result is printed.\n"
        script_content += "            # unittest.main(exit=False) returns TestProgram.\n"
        script_content += "            # But capturing success is harder without result object.\n"
        script_content += "            # Simpler fallback: just assume if it didn't exit, it passed? No.\n"
        script_content += "            # Actually, standard unittest doesn't return status clearly in main().\n"
        script_content += "            # Let's trust pytest is available in SWE-bench env (it is).\n"
        script_content += "            # Keep unittest as fallback but maybe just print nothing or minimal?\n"
        script_content += "            # Or use result = unittest.TextTestRunner().run(unittest.TestLoader().loadTestsFromModule(sys.modules[__name__]))\n"
        script_content += "        except SystemExit as e:\n"
        script_content += "             if e.code != 0:\n"
        script_content += "                 print('Issue reproduced')\n"
        script_content += "             else:\n"
        script_content += "                 print('Issue resolved')\n"
        script_content += "             sys.exit(e.code)\n"
        # Fallback for unittest main if it doesn't raise SystemExit (exit=False usage above logic was messy)
        # Revert to standard unittest.main() but wrap it?
        # Actually in SWE-bench, pytest is almost always used.
        # I'll stick to a simpler unittest block that just runs main().
        # But user wants prints.
        # Let's try:
        script_content += "        res = unittest.main(exit=False)\n"
        script_content += "        if not res.result.wasSuccessful():\n"
        script_content += "             print('Issue reproduced')\n"
        script_content += "             sys.exit(1)\n"
        script_content += "        else:\n"
        script_content += "             print('Issue resolved')\n"
        script_content += "             sys.exit(0)\n"
    
    return script_content


def identify_new_test_methods(old_content, new_content):
    """
    Parses old and new content to find newly added OR modified functions/methods that start with 'test_'.
    Priority: Added > Modified.
    Returns the name of the target test function found, or None.
    """
    import ast
    
    def get_functions(code):
        funcs = {} # name -> node
        if not code: return funcs
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef) and node.name.startswith("test_"):
                    funcs[node.name] = node
        except:
            pass
        return funcs

    old_funcs = get_functions(old_content)
    new_funcs = get_functions(new_content)
    
    # Check for added (Priority 1)
    # We want to traverse NEW content order to pick the last one if unordered dict?
    # Actually python 3.7+ dicts preserve insertion order. AST walk order is file order.
    # So list(new_funcs.keys()) is essentially ordered.
    
    added_tests = []
    for name in new_funcs:
        if name not in old_funcs:
            added_tests.append(name)
            
    if added_tests:
        return added_tests[-1]
        
    # Check for modified (Priority 2)
    modified_tests = []
    for name in new_funcs:
        if name in old_funcs:
            # Compare AST structure (ignores comments/formatting mostly)
            # ast.dump includes line numbers unless verify=False (but that's only compile)
            # wait, ast.dump includes fields. If line numbers differ (which they will if moved), it might differ.
            # We should probably compare source or use ast.dump(node, include_attributes=False) if py3.9+
            # Fallback: compare source segment? Or just content?
            # Simpler: extraction of code via ast.get_source_segment is tricky without source.
            
            # Use unparse comparison (Python 3.9+) if available, else dump
            try:
                # normalize
                n_dump = ast.dump(new_funcs[name])
                o_dump = ast.dump(old_funcs[name])
                # This is strict. If line numbers are in dump, it fails on shift.
                # But typically ast.dump DOES NOT include line numbers by default in older python?
                # Actually it does not include lineno/col_offset unless specified?
                # Let's verify defaults. Python 3.9+ 'include_attributes=False' default is False.
                # So it should be safe for logic comparison?
                
                if n_dump != o_dump:
                    modified_tests.append(name)
            except:
                pass

    if modified_tests:
        return modified_tests[-1]

    return None

def process_item_step3_full(item_tuple):
    custom_id, response_item, step1_item, instance_data, args_repostructure_dir, args_save_dir = item_tuple

    # Status check
    if 'status_code' in response_item['response'] and response_item['response']['status_code'] != 200:
        return {'status': 'failed', 'reason': 'step2_request_failed', 'custom_id': custom_id, 'details': str(response_item['response'])}

    raw_answer = response_item['response']['body']['choices'][0]['message']['content']
    
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
    
    if "__pass" in custom_id:
        instance_id = custom_id.split("__pass")[0]
    else:
        instance_id = custom_id

    structure = get_repo_structure(instance_id, args_repostructure_dir)
    if structure is None: 
        return {'status': 'failed', 'reason': 'structure_not_found', 'custom_id': custom_id, 'instance_id': instance_id}
    
    # Reconstruct found_files from Step 1
    # Check Step 1 status
    if 'status_code' in step1_item['response'] and step1_item['response']['status_code'] != 200:
        return {'status': 'failed', 'reason': 'step1_request_failed', 'custom_id': custom_id, 'details': str(step1_item['response'])}
        
    raw_answer1 = step1_item['response']['body']['choices'][0]['message']['content']
    model_found_files = raw_answer1.strip().split("\n")
    files, classes, functions = get_full_file_paths_and_classes_and_functions(structure)
    model_found_files = [correct_file_path_in_structure(file, structure) for file in model_found_files]
    found_files = correct_file_paths(model_found_files, files)
    
    # Filter for test files only
    pred_found_files_new = []
    for file in found_files:
        if 'test' in file:
            pred_found_files_new.append(file)
    found_files = pred_found_files_new

    # Top 1
    if len(found_files) > 0:
        found_files = found_files[:1]
    else:
        return {'status': 'failed', 'reason': 'no_test_files_found', 'custom_id': custom_id, 'found_files': model_found_files}

    file_contents = get_repo_files(structure, found_files)
    # file_contentes_dict = remove_test_cases(structure, {}) # REMOVE THIS
    file_contentes_dict = {} # Start empty
    
    for file_name, content in file_contents.items():
            file_contentes_dict[file_name] = {
                'old_content': content,
                'new_content': content
            }
            
    # Generate Patch
    # Note: generate_model_patch_difflib_testwritter does NOT update dict in place, so we must do it ourselves
    apply_patches(file_contentes_dict, search_replace_text)
    
    # We can still call this if we want a unified diff patch string for some other reason, 
    # but for reproduction script we rely on the updated file_contentes_dict
    _ = generate_model_patch_difflib_testwritter(file_contentes_dict=file_contentes_dict, search_replace_text=search_replace_text)
    
    # NEW LOGIC: Generate reproduce_bug.py patch
    
    # 1. Get the target file (we assume single key file modified)
    target_file = found_files[0] if found_files else None
    
    model_patch = ""
    test_func_name = None
    
    if target_file and target_file in file_contentes_dict:
        new_content = file_contentes_dict[target_file]['new_content']
        
        # Use full file strategy as requested
        # We take the entire updated file content, fix relative imports, and ensure it runs.
        repro_script_content = make_reproduction_script_full_file(new_content, file_path=target_file)
        model_patch = create_patch_from_code(repro_script_content)
            
    else:
                return {'status': 'failed', 'reason': 'target_file_not_in_contents', 'custom_id': custom_id, 'target_file': target_file, 'available': list(file_contentes_dict.keys())}

    # Prepare output dict
    # 1. Raw/Ungrouped dict (previous format, for debugging/reference)
    raw_dict = {
        "custom_id": custom_id, 
        "instance_id": instance_id,
        "raw_answer": raw_answer,
        "search_replace_text": search_replace_text,
        "model_found_files": model_found_files,
        "found_files": found_files,
        "file_contentes_dict": file_contentes_dict, 
        # Note: file_contentes_dict has full content, potentially large. 
        # If we save here, we are good.
    }
    
    # 2. Processed dict (for final output)
    instance_dict = {
        "model_name_or_path": response_item['response']['body']['model'],
        "instance_id": instance_id,
        "test_patch": model_patch,
        # User feedback: raw_test_patch should be the content of the final reproduce_bug.py
        # Consuming tool expects markdown code block
        "raw_test_patch": f"```python\n{repro_script_content}\n```",
        "test_func_name": test_func_name,
        "original_file_content": "" 
    }
    
    # SAVE LOCALLY IN WORKER to avoid IPC
    # We need samples_dir. It is derived from args_save_dir passed in item_tuple
    import os
    import json
    samples_dir = os.path.join(args_save_dir, 'samples')
    # os.makedirs(samples_dir, exist_ok=True) # Likely already exists, or race to create?
    # makedirs is thread/process safe in recent python? Yes, with exist_ok=True.
    
    try:
        output_file_path = os.path.join(samples_dir, f'{custom_id}.jsonl')
        with open(output_file_path, 'w') as out_f:
            out_f.write(json.dumps(raw_dict, ensure_ascii=False) + '\n')
            
        return {
            'status': 'success',
            'custom_id': custom_id,
            'instance_dict_str': json.dumps(instance_dict) # Return string to minimize pickling overhead? logic in main uses string.
        }
    except Exception as e:
        return {'status': 'failed', 'reason': 'save_error', 'custom_id': custom_id, 'details': str(e)}


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
    parser_s3.add_argument('--step1_file', required=True, help="Output of Stage 1 batch (needed for context)")
    parser_s3.add_argument('--save_dir', required=True)
    parser_s3.add_argument('--num_workers', type=int, default=8, help='Number of worker processes for parallel processing')

    args = parser.parse_args()
    
    if args.stage == 'gen_step1':
        gen_step1(args)
    elif args.stage == 'proc_step1_gen_step2':
        proc_step1_gen_step2(args)
    elif args.stage == 'proc_step2':
        proc_step2(args)
