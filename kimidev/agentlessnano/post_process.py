import ast
import copy
import difflib
import os
import re
import subprocess
import uuid
from collections import OrderedDict
from difflib import unified_diff

from kimidev.agentlessnano.utils import *


def check_syntax(code):
    if not isinstance(code, list):
        code = [code]

    for c in code:
        if not c.strip():  # Check for cases where the model didn't return a python block
            return False
        try:
            ast.parse(c)
        except SyntaxError:
            return False
    return True


def remove_empty_lines(code: str) -> str:
    # Split the code into lines
    lines = code.splitlines()
    # Remove empty lines
    filtered_lines = [line for line in lines if line.strip() != '']
    return '\n'.join(filtered_lines)


def check_code_differ_by_just_empty_lines(codes, prev_codes) -> bool:
    if not isinstance(codes, list):
        codes = [codes]
        prev_codes = [prev_codes]

    normalized_code1 = ''
    normalized_code2 = ''

    for code, prev_code in zip(codes, prev_codes, strict=False):
        # Normalize both code snippets
        normalized_code1 += remove_empty_lines(code)
        normalized_code2 += remove_empty_lines(prev_code)

    return normalized_code1 == normalized_code2


def lint_code(repo_playground, temp_name, code, prev_code='') -> tuple[bool, set, set]:
    # Generate a temperary folder and add uuid to avoid collision
    repo_playground = os.path.join(repo_playground, str(uuid.uuid4()))

    # assert playground doesn't exist
    assert not os.path.exists(repo_playground), f'{repo_playground} already exists'

    # create playground
    os.makedirs(repo_playground)

    with open(f'{repo_playground}/{temp_name}', 'w') as f:
        f.write(prev_code)

    # lint the code
    # check for fatal errors
    fatal = 'E9,F821,F823,F831,F406,F407,F701,F702,F704,F706'
    o = subprocess.run(
        f'flake8 --select={fatal} --isolated {repo_playground}/{temp_name}',
        shell=True,
        capture_output=True,
    )
    s = o.stdout.decode('utf-8')

    prev_errors = set()
    if s != '':
        for error in s.split(f'{repo_playground}/{temp_name}:')[1:]:
            num_free_error = ':'.join(error.split(':')[2:]).strip()
            prev_errors.add(num_free_error)

    with open(f'{repo_playground}/{temp_name}', 'w') as f:
        f.write(code)

    o = subprocess.run(
        f'flake8 --select={fatal} --isolated {repo_playground}/{temp_name}',
        shell=True,
        capture_output=True,
    )
    s = o.stdout.decode('utf-8')

    # remove playground
    subprocess.run(f'rm -rf {repo_playground}', shell=True)

    errors = set()
    if s != '':
        for error in s.split(f'{repo_playground}/{temp_name}:')[1:]:
            num_free_error = ':'.join(error.split(':')[2:]).strip()
            errors.add(num_free_error)

    if len(errors - prev_errors) > 0:
        return False, prev_errors, errors

    return True, set(), set()


def fake_git_repo(repo_playground, file_pathes, old_contents, new_contents) -> str:
    """create a fake git repo to obtain git diff format"""

    if not isinstance(file_pathes, list):
        # for backwards compatibility
        file_pathes = [file_pathes]
        old_contents = [old_contents]
        new_contents = [new_contents]

    # Generate a temperary folder and add uuid to avoid collision
    repo_playground = os.path.join(repo_playground, str(uuid.uuid4()))

    # assert playground doesn't exist
    assert not os.path.exists(repo_playground), f'{repo_playground} already exists'

    # create playground
    os.makedirs(repo_playground)

    # create a fake git repo
    subprocess.run(f'cd {repo_playground} && git init', shell=True)

    for file_path, old_content, new_content in zip(
        file_pathes,
        old_contents,
        new_contents,
        strict=False,
    ):
        # create a file
        subprocess.run(
            f'mkdir -p {repo_playground}/{os.path.dirname(file_path)}',
            shell=True,
        )

        with open(f'{repo_playground}/{file_path}', 'w') as f:
            f.write(old_content)

        # add file to git
        # same message is okay
        subprocess.run(
            f"cd {repo_playground} && git add {file_path} && git commit -m 'initial commit'",
            shell=True,
        )

    for file_path, old_content, new_content in zip(
        file_pathes,
        old_contents,
        new_contents,
        strict=False,
    ):
        # edit file
        with open(f'{repo_playground}/{file_path}', 'w') as f:
            f.write(new_content)

    # get git diff
    o = subprocess.run(
        f'cd {repo_playground} && git diff .',
        shell=True,
        capture_output=True,
    )

    s = o.stdout.decode('utf-8')

    # remove playground
    subprocess.run(f'rm -rf {repo_playground}', shell=True)

    return s


def fake_git_apply(repo_playground, file_path, old_content, patch) -> str:
    """create a fake git repo to obtain new file content"""

    # Generate a temperary folder and add uuid to avoid collision
    repo_playground = os.path.join(repo_playground, str(uuid.uuid4()))

    # assert playground doesn't exist
    assert not os.path.exists(repo_playground), f'{repo_playground} already exists'

    # create playground
    os.makedirs(repo_playground)

    # create a fake git repo
    subprocess.run(f'cd {repo_playground} && git init', shell=True)

    # create a file
    subprocess.run(
        f'mkdir -p {repo_playground}/{os.path.dirname(file_path)}',
        shell=True,
    )

    with open(f'{repo_playground}/{file_path}', 'w') as f:
        f.write(old_content)

    # add file to git
    subprocess.run(
        f"cd {repo_playground} && git add {file_path} && git commit -m 'initial commit'",
        shell=True,
    )

    # apply patch file
    patch_file = f'{uuid.uuid4()!s}.patch'
    with open(f'{repo_playground}/{patch_file}', 'w') as f:
        f.write(patch)
    o = subprocess.run(
        f'cd {repo_playground} && git apply --whitespace=nowarn {patch_file}',
        shell=True,
        capture_output=True,
    )
    if o.stderr.decode('utf-8'):
        print('stderr> ', o.stderr.decode('utf-8'))
        # TODO: This rarely happen but the patch should be valid, needs to look into it

        with open(f'{repo_playground}/{file_path}', 'w') as f:
            f.write(old_content + '\n')

        o = subprocess.run(
            f'cd {repo_playground} && git apply --whitespace=nowarn {patch_file}',
            shell=True,
            capture_output=True,
        )

        if o.stderr.decode('utf-8'):
            print('stderr> ', o.stderr.decode('utf-8'))
            assert False, "shouldn't happen"

    # get git diff
    o = subprocess.run(
        f'cd {repo_playground} && cat {file_path}',
        shell=True,
        capture_output=True,
    )

    s = o.stdout.decode('utf-8')

    # remove playground
    subprocess.run(f'rm -rf {repo_playground}', shell=True)

    return s


def fake_git_apply_multiple(repo_playground, file_path_contents, patch) -> dict:
    """create a fake git repo to obtain new file contents (multiple)"""

    # Generate a temperary folder and add uuid to avoid collision
    repo_playground = os.path.join(repo_playground, str(uuid.uuid4()))

    # assert playground doesn't exist
    assert not os.path.exists(repo_playground), f'{repo_playground} already exists'

    # create playground
    os.makedirs(repo_playground)

    # create a fake git repo
    subprocess.run(f'cd {repo_playground} && git init', shell=True)

    # create files
    for file_path, old_content in file_path_contents.items():
        subprocess.run(
            f'mkdir -p {repo_playground}/{os.path.dirname(file_path)}',
            shell=True,
        )

        with open(f'{repo_playground}/{file_path}', 'w') as f:
            f.write(old_content)

        # add file to git
        subprocess.run(
            f"cd {repo_playground} && git add {file_path} && git commit -m 'initial commit'",
            shell=True,
        )

    # apply patch file
    patch_file = f'{uuid.uuid4()!s}.patch'
    with open(f'{repo_playground}/{patch_file}', 'w') as f:
        f.write(patch)
    o = subprocess.run(
        f'cd {repo_playground} && git apply --whitespace=nowarn {patch_file}',
        shell=True,
        capture_output=True,
    )
    if o.stderr.decode('utf-8'):
        print('stderr> ', o.stderr.decode('utf-8'))
        # TODO: This rarely happen but the patch should be valid, needs to look into it
        for file_path, old_content in file_path_contents.items():
            with open(f'{repo_playground}/{file_path}', 'w') as f:
                f.write(old_content + '\n')

        o = subprocess.run(
            f'cd {repo_playground} && git apply --whitespace=nowarn {patch_file}',
            shell=True,
            capture_output=True,
        )

        if o.stderr.decode('utf-8'):
            print('stderr> ', o.stderr.decode('utf-8'))
            assert False, "shouldn't happen"

    new_file_path_contents = {}

    # get git diff
    for file_path, old_content in file_path_contents.items():
        o = subprocess.run(
            f'cd {repo_playground} && cat {file_path}',
            shell=True,
            capture_output=True,
        )

        s = o.stdout.decode('utf-8')

        new_file_path_contents[file_path] = s

    # remove playground
    subprocess.run(f'rm -rf {repo_playground}', shell=True)

    return new_file_path_contents


def get_functions(tree):
    """Get a set of function and method names from the AST tree."""
    functions = {}

    class FunctionVisitor(ast.NodeVisitor):
        def __init__(self):
            self.parents = []

        def visit(self, node):
            self.parents.append(node)
            super().visit(node)
            self.parents.pop()

        def visit_FunctionDef(self, node):
            if not any(isinstance(parent, ast.ClassDef) for parent in self.parents):
                functions[node.name] = ast.unparse(node)
            self.generic_visit(node)

        def visit_AsyncFunctionDef(self, node):
            if not any(isinstance(parent, ast.ClassDef) for parent in self.parents):
                functions[node.name] = ast.unparse(node)
            self.generic_visit(node)

    class ClassVisitor(ast.NodeVisitor):
        def visit_ClassDef(self, node):
            class_name = node.name
            for body_item in node.body:
                if isinstance(body_item, ast.FunctionDef) or isinstance(
                    body_item,
                    ast.AsyncFunctionDef,
                ):
                    functions[f'{class_name}.{body_item.name}'] = ast.unparse(body_item)
            self.generic_visit(node)

    FunctionVisitor().visit(tree)
    ClassVisitor().visit(tree)
    return functions


def is_just_new_function(code1, code2):
    tree1 = ast.parse(code1)
    tree2 = ast.parse(code2)

    functions1 = get_functions(tree1)
    functions2 = get_functions(tree2)

    # The new functions in the second code
    if len(set(list(functions1.keys())) - set(list(functions2.keys()))) > 0:
        # removes functions
        return False

    for func in functions1:
        if functions1[func] != functions2[func]:
            # modifies existing functions
            return False

    if len(set(list(functions2.keys())) - set(list(functions1.keys()))) > 0:
        return True

    # modifying global stuff is okay, because its actually same as functions almost.

    return False


import io
import tokenize


def remove_comments_and_docstrings(source):
    io_obj = io.StringIO(source)
    out = ''
    prev_toktype = tokenize.INDENT
    last_lineno = -1
    last_col = 0
    for tok in tokenize.generate_tokens(io_obj.readline):
        token_type = tok[0]
        token_string = tok[1]
        start_line, start_col = tok[2]
        end_line, end_col = tok[3]
        ltext = tok[4]
        if start_line > last_lineno:
            last_col = 0
        if start_col > last_col:
            out += ' ' * (start_col - last_col)
        if token_type == tokenize.COMMENT:
            pass
        elif token_type == tokenize.STRING:
            if prev_toktype != tokenize.INDENT:
                if prev_toktype != tokenize.NEWLINE:
                    if start_col > 0:
                        out += token_string
        else:
            out += token_string
        prev_toktype = token_type
        last_col = end_col
        last_lineno = end_line
    out = '\n'.join(l for l in out.splitlines() if l.strip())
    return out


def normalize_patch(
    instance_id,
    patch: str,
    original_file_content: list,
    new_file_content: list,
    edited_files: list,
) -> str:
    "Remove edits to trailing spaces and comments in the patch."
    if not patch.strip():
        return ''

    normalized_diff = ''

    for o_file_content, n_file_content, edited_file in zip(
        original_file_content,
        new_file_content,
        edited_files,
        strict=False,
    ):
        old_content = o_file_content
        new_content = n_file_content

        # Normalize file contents
        def normalize_code(code):
            try:
                node = ast.parse(code)
                return ast.unparse(node)
            except:
                return code

        old_content = normalize_code(old_content)
        new_content = normalize_code(new_content)

        try:
            remove_docstring_old_content = remove_comments_and_docstrings(old_content)
            ast.parse(remove_docstring_old_content)  # check
            remove_docstring_new_content = remove_comments_and_docstrings(new_content)
            ast.parse(remove_docstring_new_content)  # check
        except:
            # when does this exception happen?
            # when the code has some class or function with empty docstring (thats valid python code)
            # but removing it is not, to be save we just use the original.
            remove_docstring_old_content = old_content
            remove_docstring_new_content = new_content

        diff = fake_git_repo(
            'playground',
            edited_file,
            remove_docstring_old_content,
            remove_docstring_new_content,
        )

        if is_just_new_function(
            remove_docstring_old_content,
            remove_docstring_new_content,
        ):
            # modify the diff to ignore context.
            new_diff = []
            for line in diff.splitlines():
                if line.startswith('-') or line.startswith('+'):
                    new_diff.append(line)
            diff = '\n'.join(new_diff)

        normalized_diff += diff

    # Note that the normalized diff may not be applied to the original file.
    return normalized_diff


def extract_python_blocks(text):
    # Regular expression pattern to match ```python\n{text}\n```
    pattern = r'```python\n(.*?)\n```'

    # Use re.findall to find all matches
    matches = re.findall(pattern, text, re.DOTALL)

    return matches


def extract_code_blocks(text):
    pattern = r'```\n(.*?)\n```'
    matches = re.findall(pattern, text, re.DOTALL)
    if len(matches) == 0:
        if '```' in text:
            # handle the case where the code block is not complete
            return [text.split('```', 1)[-1].strip()]
    return matches


def extract_locs_for_files(locs, file_names, keep_old_order=False):
    if keep_old_order:
        results = {fn: [] for fn in file_names}
    else:
        results = {}  # dict is insertion ordered
    current_file_name = None
    for loc in locs:
        for line in loc.splitlines():
            if line.strip().endswith('.py'):
                current_file_name = line.strip()
            elif line.strip() and any(
                line.startswith(w) for w in ['line:', 'function:', 'class:', 'variable:']
            ):
                if current_file_name in file_names:
                    if current_file_name not in results:
                        results[current_file_name] = []
                    results[current_file_name].append(line)
                else:
                    pass

    for file_name in file_names:
        if file_name not in results:  # guard for new order case
            results[file_name] = []

    return {fn: ['\n'.join(results[fn])] for fn in results.keys()}


def extract_starting_number(subcommand):
    return int(subcommand.split(',')[0].split('start=')[-1])


def extract_ending_number(subcommand):
    return int(subcommand.split(',')[1].split('end=')[-1])


def overlap(subcommand1, subcommand2):
    start1, end1 = (
        extract_starting_number(subcommand1),
        extract_ending_number(
            subcommand1,
        ),
    )
    start2, end2 = (
        extract_starting_number(subcommand2),
        extract_ending_number(
            subcommand2,
        ),
    )
    return not (end1 < start2 or end2 < start1)


def split_edit_multifile_commands(
    commands,
    diff_format=False,
    str_replace_format=False,
) -> dict[str, str]:
    """Split commands based on edited files."""
    file_to_commands = OrderedDict()
    if str_replace_format:
        for command in commands:
            file_name = None
            for json_message in command:
                if json_message['type'] == 'tool_use':
                    str_replace_command = json_message['input']

                    if 'command' not in str_replace_command:
                        str_replace_command['command'] = 'str_replace'

                    if 'path' not in str_replace_command:
                        continue

                    if 'command' not in str_replace_command:
                        continue

                    if str_replace_command['command'] == 'str_replace':
                        if 'old_str' not in str_replace_command:
                            continue

                        if 'new_str' not in str_replace_command:
                            # add empty string
                            str_replace_command['new_str'] = ''

                    if str_replace_command['command'] == 'insert':
                        if (
                            'insert_line' not in str_replace_command
                            or 'new_str' not in str_replace_command
                        ):
                            continue

                    file_name = "'" + str_replace_command['path'] + "'"
                    # deduplicate
                    if (
                        file_name not in file_to_commands
                        or str_replace_command not in file_to_commands[file_name]
                    ):
                        file_to_commands.setdefault(file_name, []).append(
                            str_replace_command,
                        )

    elif diff_format:
        for command in commands:
            file_name = None
            for subcommand in command.split('>>>>>>> REPLACE')[:-1]:
                subcommand = subcommand.strip()
                if '<<<<<<< SEARCH' in subcommand:
                    fn = subcommand.split('<<<<<<< SEARCH')[0].lstrip('#').strip()
                    if fn:
                        file_name = "'" + fn + "'"

                if len(subcommand.split('<<<<<<< SEARCH')) != 2:
                    continue
                converted_command = (
                    '<<<<<<< SEARCH'
                    + subcommand.split('<<<<<<< SEARCH')[1]
                    + '\n'
                    + '>>>>>>> REPLACE'
                )
                # deduplicate
                if (
                    file_name not in file_to_commands
                    or converted_command not in file_to_commands[file_name]
                ):
                    file_to_commands.setdefault(file_name, []).append(converted_command)

    else:
        for command in commands:
            for subcommand in command.split('edit_file(')[1:]:
                file_name, start, end, content = subcommand.split(',', 3)
                converted_command = 'edit_file(' + ','.join([start, end, content])
                # deduplicate
                if (
                    file_name not in file_to_commands
                    or converted_command not in file_to_commands[file_name]
                ):
                    file_to_commands.setdefault(file_name, []).append(converted_command)

    return file_to_commands


def parse_str_replace_edit_commands(
    commands,
    content,
    file_loc_intervals: list[tuple[int, int]],
):
    """
    Original implementation: https://github.com/anthropics/anthropic-quickstarts/blob/main/computer-use-demo/computer_use_demo/tools/edit.py
    """
    # let's first make sure the intervals are sorted
    file_loc_intervals.sort()
    replaced = False
    # apply the edits from the end of file to the beginning of file
    # this is to make sure context is correct
    for interval in file_loc_intervals[::-1]:
        start, end = interval
        start = max(start - 1, 0)
        # intervals are 1-indexed

        context_segment = '\n'.join(content.splitlines()[start:end])
        context_segment = '\n' + context_segment + '\n'

        # since we want to replace the original context, let's first check for all edits.
        can_apply = []
        for subcommand in commands:
            if subcommand['command'] == 'str_replace':
                original, replace = subcommand['old_str'], subcommand['new_str']

                if original in context_segment:
                    can_apply.append(subcommand)

            elif subcommand['command'] == 'insert':
                insert_line = subcommand['insert_line']
                # TODO check this
                if insert_line > start and insert_line <= end:
                    can_apply.append(subcommand)

        # apply edits backwards
        for subcommand in can_apply[::-1]:
            if subcommand['command'] == 'str_replace':
                original, replace = subcommand['old_str'], subcommand['new_str']

                if (
                    original in context_segment
                ):  # This may not be true after some previously applied edits
                    context_segment = context_segment.replace(original, replace)
                    replaced = True

            elif subcommand['command'] == 'insert':
                insert_line = subcommand['insert_line']
                replace = subcommand['new_str']
                # TODO check this
                if insert_line > start and insert_line <= end:
                    if insert_line != start - 1:
                        replace = '\n' + replace
                    if insert_line != end:
                        replace = replace + '\n'
                    context_segment = (
                        '\n'.join(
                            context_segment.splitlines()[: insert_line - start + 1],
                        )
                        + replace
                        + '\n'.join(
                            context_segment.splitlines()[insert_line - start + 1 :],
                        )
                    )
                    context_segment = context_segment + '\n'
                    replaced = True

        # reassembly
        content = (
            '\n'.join(content.splitlines()[:start])
            + context_segment
            + '\n'.join(content.splitlines()[end:])
        )

    if not replaced:
        print('not replaced')

    return content


def parse_diff_edit_commands(
    commands,
    content,
    file_loc_intervals: list[tuple[int, int]],
):
    def parse_for_threedots(original, replace, file_loc_intervals, content):
        # if dot dot dot in replace, its always safe to remove it
        if replace.startswith('...\n') and len(replace) > 4:
            # im parsing this first because its used later on
            replace = replace[4:]

        # just dot dot dot, then need to do something special
        if original == '...':
            if not replace[0].isspace():
                # this is okay
                # find a suitable original string to replace
                for interval in file_loc_intervals:
                    start, end = interval
                    start = max(start - 1, 0)
                    context_segment = '\n'.join(content.splitlines()[start:end])

                    for line in context_segment.splitlines():
                        if len(line) > 0 and not line[0].isspace():
                            if content.count(line) == 1:
                                original = line
                                # keep the line
                                replace = replace + '\n\n' + line
                                break

                    if original != '...':
                        break

                if original == '...':
                    print('cannot find suitable location')

        # dot dot dot with something else in original, then its safe to replace
        if original.startswith('...\n') and len(original) > 4:
            # remove dot dot dot from original
            original = original[4:]

        return original, replace

    if len(file_loc_intervals) == 0:
        if original in content:
            content = content.replace(original, replace)
            replaced = True
        else:
            print('not replaced')
        return content
    # let's first make sure the intervals are sorted
    file_loc_intervals.sort()
    replaced = False
    # apply the edits from the end of file to the beginning of file
    # this is to make sure context is correct
    for interval in file_loc_intervals[::-1]:
        start, end = interval
        start = max(start - 1, 0)
        context_segment = '\n'.join(content.splitlines()[start:end])
        context_segment = '\n' + context_segment + '\n'

        # since we want to replace the original context, let's first check for all edits.
        can_apply = []
        for subcommand in commands:
            if not subcommand.startswith('<<<<<<< SEARCH') and subcommand.endswith(
                '>>>>>>> REPLACE',
            ):
                continue

            subcommand = '\n'.join(subcommand.splitlines()[1:-1])
            if len(subcommand.split('\n=======\n')) != 2:
                continue

            original, replace = subcommand.split('\n=======\n')

            original, replace = parse_for_threedots(
                original,
                replace,
                file_loc_intervals,
                content,
            )

            original = '\n' + original + '\n'
            replace = '\n' + replace + '\n'

            if original in context_segment:
                can_apply.append(subcommand)

        # apply edits backwards
        for subcommand in can_apply[::-1]:
            original, replace = subcommand.split('\n=======\n')

            original, replace = parse_for_threedots(
                original,
                replace,
                file_loc_intervals,
                content,
            )

            original = '\n' + original + '\n'
            replace = '\n' + replace + '\n'
            if (
                original in context_segment
            ):  # This may not be true after some previously applied edits
                context_segment = context_segment.replace(original, replace)
                replaced = True
        # reassembly
        content = (
            '\n'.join(content.splitlines()[:start])
            + context_segment
            + '\n'.join(content.splitlines()[end:])
        )

    if not replaced:
        print('not replaced')

    return content


def parse_edit_commands(commands, content):
    content_lines = content.splitlines()
    map_content_lines_to_line_num = [i for i in range(len(content_lines) + 1)]

    # Make a list of the subcommands
    subcommands = []

    for command in commands:
        for subcommand in command.split('edit_file(')[1:]:
            subcommands.append(subcommand)

    # Remove duplicates while preserving order
    seen = set()
    unique_subcommands = []
    for subcommand in subcommands:
        if subcommand not in seen:
            unique_subcommands.append(subcommand)
            seen.add(subcommand)

    # Sort the unique subcommands by the starting number in reverse order
    unique_sorted_subcommands = sorted(
        unique_subcommands,
        key=extract_starting_number,
        reverse=True,
    )

    for subcommand in unique_sorted_subcommands:
        command_start = int(subcommand.split(',')[0].split('start=')[-1])
        command_end = int(subcommand.split(',')[1].split('end=')[-1])

        start = map_content_lines_to_line_num.index(command_start)
        end = map_content_lines_to_line_num.index(command_end)

        try:
            changed_content = eval(
                ')'.join(','.join(subcommand.split(',')[2:]).split(')')[:-1]),
            )
            # small thing to ensure right white space indent
            if (
                start == end
                and len(changed_content.splitlines()) == 1
                and (len(changed_content) - len(changed_content.lstrip())) == 0
            ):
                indent_length = len(content_lines[start - 1]) - len(
                    content_lines[start - 1].lstrip(),
                )
                changed_content = ' ' * indent_length + changed_content.lstrip()

        # catch syntax error
        except:
            # try to fix, specially for case where there are """ or ''' in the string, that cannot be
            # easily evaluated.
            eval_str = ')'.join(
                ','.join(subcommand.split(',')[2:]).split(')')[:-1],
            ).strip()

            eval_str = eval_str.removeprefix('content=')

            if eval_str.startswith('"""') or eval_str.startswith("'''"):
                eval_str = eval_str[3:-3]
            if eval_str.startswith('"') or eval_str.startswith("'"):
                eval_str = eval_str[1:-1]

            changed_content = eval_str

        content_lines[start - 1 : end] = changed_content.splitlines()

    content = '\n'.join(content_lines)

    return content


def test_parse_str_replace():
    content = """A
B
C
D
E
F
""".strip()

    from agentless.util.preprocess_data import line_wrap_content

    shown_content = line_wrap_content(content, context_intervals=[(3, 5)])
    print(shown_content)

    commands = [
        {'command': 'insert', 'path': 'test', 'insert_line': 1, 'new_str': 'test_str'},
    ]
    new_content = parse_str_replace_edit_commands(
        commands,
        content,
        file_loc_intervals=[(1, 5)],
    )
    print(new_content)


def test_parse():
    raw_output = """
```python
edit_file(1, 1, "import os")
```
"""

    content = """
import sys
""".strip()

    commands = extract_python_blocks(raw_output)

    content = parse_edit_commands(commands, content)

    assert content == 'import os', content

    raw_output = """
```python
edit_file(1, 1, '''import os\nimport sys''')
```
"""

    content = """
import sys
""".strip()

    commands = extract_python_blocks(raw_output)

    content = parse_edit_commands(commands, content)

    assert content == 'import os\nimport sys', content

    raw_output = """
```python
edit_file(1, 1, '''import os''')
edit_file(1, 1, '''import sys''')
```
"""

    content = """
import sys
""".strip()

    commands = extract_python_blocks(raw_output)

    content = parse_edit_commands(commands, content)

    assert content == 'import sys', content

    content = """
test
testing
""".strip()
    raw_output = """
```python
edit_file(1, 1, "testing\ntesting2")
edit_file(2, 2, "testing3")
```
"""
    content = parse_edit_commands(extract_python_blocks(raw_output), content)

    assert content == 'testing\ntesting2\ntesting3', content

    content = """
test
testing
testinging
""".strip()

    raw_output = """
```python
edit_file(1, 2, "testing")
edit_file(3, 3, "testing3")
```
"""
    content = parse_edit_commands(extract_python_blocks(raw_output), content)

    assert content == 'testing\ntesting3', content

    content = """
test
testing
testinging
test
testing
""".strip()

    raw_output = """
```python
edit_file(1, 2, "testing")
edit_file(3, 3, "testing3")
edit_file(4, 4, "testing\ntesting2")
edit_file(5, 5, "testing3")
```
"""

    edited_content = """
testing
testing3
testing
testing2
testing3
""".strip()
    content = parse_edit_commands(extract_python_blocks(raw_output), content)

    assert content == edited_content, content

    # Test for if command is present twice in the output (adapted from real output)
    content = """
test
testing
testinging
test
testing
""".strip()

    raw_output = (
        """
"To fix the issue where `viewcode`
1. Add a condition

```python
edit_file(4, 4, """
        """
test4
"""
        """)
```

2. Ensure that the `doctree_read`

```python
edit_file(2, 3, """
        """
test-2-3
"""
        """)
```

Here is the complete set of commands to address the issue:

```python
edit_file(4, 4, """
        """
test4
"""
        """)
```

```python
edit_file(2, 3, """
        """
test-2-3
"""
        """)
```
"""
    )
    content = parse_edit_commands(extract_python_blocks(raw_output), content)

    revised_content = """
test
test-2-3
test4
testing
""".strip()

    assert content == revised_content, content

    raw_output = """
```
django/db/migrations/optimizer.py
function: MigrationOptimizer.optimize_inner

django/db/migrations/operations/fields.py
function: AlterField.reduce
```
"""
    files = [
        'django/db/migrations/optimizer.py',
        'django/db/migrations/operations/fields.py',
        'django/db/migrations/operations/models.py',
    ]
    extracted_locs = extract_locs_for_files([raw_output], files)
    print(extracted_locs)
    assert extracted_locs == [
        ['function: MigrationOptimizer.optimize_inner'],
        ['function: AlterField.reduce'],
        [''],
    ]

    raw_output = """
```python
edit_file(start=1, end=1, content="testing not")
```
"""

    content = """
testing
""".strip()

    edited_content = """
testing not
""".strip()

    content = parse_edit_commands(extract_python_blocks(raw_output), content)
    assert content == edited_content, edited_content


def _post_process_multifile_repair_rmlogger(
    raw_output: str,
    file_contents: dict[str, str],
    file_loc_intervals: dict[str, list],
    diff_format=False,
):
    edit_multifile_commands = extract_python_blocks(raw_output)

    edited_file = ''
    new_content = ''
    try:
        file_to_commands = split_edit_multifile_commands(
            edit_multifile_commands,
            diff_format=diff_format,
        )
        # Let's only edit the first file in the edit commands.
        edited_file_key = next(iter(file_to_commands.keys()))
        edit_commands = file_to_commands[edited_file_key]
        edited_file = eval(edited_file_key)  # convert '"file.py"' to 'file.py'
        content = file_contents[edited_file]
        if diff_format:
            new_content = parse_diff_edit_commands(
                edit_commands,
                content,
                file_loc_intervals[edited_file],
            )
        else:
            new_content = parse_edit_commands(edit_commands, content)
    except Exception:
        return edited_file, new_content

    diff = list(
        unified_diff(
            content.split('\n'),
            new_content.split('\n'),
            fromfile=edited_file,
            tofile=edited_file,
            lineterm='',
        ),
    )
    # print("\n".join(diff))
    return edited_file, new_content


def post_process_raw_output_wolog(
    raw_output_text,
    file_contents,
    logger,
    file_loc_intervals,
    args=None,
):
    git_diffs = ''
    raw_git_diffs = ''
    lint_success = False
    content = ''
    try:
        edited_file, new_content = _post_process_multifile_repair_rmlogger(
            raw_output_text,
            file_contents,
            file_loc_intervals,
            diff_format=True,
        )
        if edited_file in file_contents:
            content = file_contents[edited_file]

            git_diff = fake_git_repo('playground', edited_file, content, new_content)

            raw_git_diffs += '\n' + git_diff.replace(
                '\\ No newline at end of file\n',
                '',
            )

            syntax_success = check_syntax(new_content)
            lint_success, prev_errors, errors = lint_code(
                'playground',
                'test.py',
                new_content,
                file_contents[edited_file],
            )

            differ_by_empty_lines = check_code_differ_by_just_empty_lines(
                new_content,
                file_contents[edited_file],
            )

            print(lint_success, prev_errors, errors, differ_by_empty_lines)

            if syntax_success and not differ_by_empty_lines:
                git_diffs = raw_git_diffs
            else:
                git_diffs = ''  # no need to evaluate
        else:
            diff = list(
                unified_diff(
                    content.split('\n'),
                    new_content.split('\n'),
                    fromfile=edited_file,
                    tofile=edited_file,
                    lineterm='',
                ),
            )
            print('Failed parsing diff!')
            print('\n'.join(diff))
    except Exception as e:
        print(raw_output_text)
        print(e)

    return git_diffs, raw_git_diffs, content


def generate_model_patch(
    structure,
    found_files,
    found_edit_locs,
    pred_found_files,
    search_replace_text,
    context_window=10,
    loc_interval=True,
    fine_grain_loc_only=False,
    add_space=False,
    no_line_number=True,
    sticky_scroll=False,
):
    """
    Generate a model patch by constructing the top-N file context, processing raw output,
    and formatting the Git diff.

    Args:
        structure (dict): The repository structure.
        found_files (list): List of found files.
        found_edit_locs (list): List of edit locations.
        pred_found_files (list): List of predicted found files.
        search_replace_text (str): Text for search and replace.
        context_window (int, optional): Context window size. Defaults to 10.
        loc_interval (bool, optional): Whether to include location intervals. Defaults to True.
        fine_grain_loc_only (bool, optional): Fine-grained location only. Defaults to False.
        add_space (bool, optional): Whether to add spaces in processing. Defaults to False.
        no_line_number (bool, optional): Exclude line numbers. Defaults to True.
        sticky_scroll (bool, optional): Enable sticky scroll. Defaults to False.

    Returns:
        str: The generated model patch.
    """
    if not isinstance(found_edit_locs, list):
        found_edit_locs = [found_edit_locs]

    file_contents = get_repo_files(structure, found_files)
    found_edit_locs_dict = {k: v for item in found_edit_locs for k, v in item.items()}

    topn_content, file_loc_intervals = construct_topn_file_context(
        found_edit_locs_dict,
        pred_found_files,
        file_contents,
        structure,
        context_window=context_window,
        loc_interval=loc_interval,
        fine_grain_loc_only=fine_grain_loc_only,
        add_space=add_space,
        no_line_number=no_line_number,
        sticky_scroll=sticky_scroll,
    )

    # Post process output
    git_diffs, raw_git_diffs, content = post_process_raw_output_wolog(
        search_replace_text,
        file_contents,
        None,
        file_loc_intervals,
    )

    return git_diffs.lstrip()


def generate_model_patch_difflib(structure, search_replace_text):
    def pre_process_SearPlace(search_replace_texts):
        # Updated regex pattern to account for possible code block formatting
        # ([^\n]+): This captures the file name, ensuring it only matches non-newline characters (so it won't include unwanted newlines).
        pattern = r'(?:### )?([^\n]+)\n<{4,} SEARCH\n(.*?)\n={4,}\n(.*?)\n>{4,} REPLACE'
        # pattern = r'### ([^\n]+)\n<<<<<<< SEARCH\n(.*?)\n=======\n(.*?)\n>>>>>>> REPLACE'

        # Find all matches
        matches = re.findall(pattern, search_replace_texts, re.DOTALL)

        # Extract file names, search strings, and replace strings from matches
        file_names = [match[0] for match in matches]
        search_strings = [match[1] for match in matches]
        replace_strings = [match[2] for match in matches]

        return file_names, search_strings, replace_strings

    def update_file_content(file_names, search_strings, replace_strings, content_dict):
        new_content_dict = copy.deepcopy(content_dict)
        for file_name, search_str, replace_str in zip(
            file_names,
            search_strings,
            replace_strings,
            strict=False,
        ):
            if file_name in content_dict:  # Perform the replacement in the content
                new_content_dict[file_name] = new_content_dict[file_name].replace(
                    search_str,
                    replace_str,
                )
        return new_content_dict

    file_names, search_strings, replace_strings = pre_process_SearPlace(search_replace_text)
    file_names = [correct_file_path_in_structure(file, structure) for file in file_names]
    ori_file_contents = get_repo_files(structure, file_names)

    new_file_contents = update_file_content(
        file_names,
        search_strings,
        replace_strings,
        ori_file_contents,
    )

    stage2_edits = {
        file: [ori_file_contents.get(file, ''), new_file_contents.get(file, '')]
        for file in file_names
    }

    file_level_diff = ''
    for fn, edit_list in stage2_edits.items():
        now_diff = difflib.unified_diff(
            edit_list[0].splitlines(),
            edit_list[-1].splitlines(),
            lineterm='',
        )
        now_diff_content = f'diff --git a/{fn} b/{fn}\n'
        for diff_line in now_diff:
            if diff_line == '--- ':
                now_diff_content += diff_line + f'a/{fn}\n'
            elif diff_line == '+++ ':
                now_diff_content += diff_line + f'b/{fn}\n'
            else:
                now_diff_content += diff_line + '\n'

        file_level_diff += now_diff_content

    return file_level_diff

def generate_model_patch_difflib_testwritter(file_contentes_dict, search_replace_text):
    def pre_process_SearPlace(search_replace_texts):
        # Updated regex pattern to account for possible code block formatting
        # ([^\n]+): This captures the file name, ensuring it only matches non-newline characters (so it won't include unwanted newlines).
        pattern = r'(?:### )?([^\n]+)\n<{4,} SEARCH\n(.*?)\n={4,}\n(.*?)\n>{4,} REPLACE'
        # pattern = r'### ([^\n]+)\n<<<<<<< SEARCH\n(.*?)\n=======\n(.*?)\n>>>>>>> REPLACE'
        
        # Find all matches
        matches = re.findall(pattern, search_replace_texts, re.DOTALL)
        
        # Extract file names, search strings, and replace strings from matches
        file_names = [match[0] for match in matches] 
        search_strings = [match[1] for match in matches] 
        replace_strings = [match[2] for match in matches]

        return file_names, search_strings, replace_strings

    def update_file_content(file_names, search_strings, replace_strings, content_dict): 
        new_content_dict = copy.deepcopy(content_dict)
        for file_name, search_str, replace_str in zip(file_names, search_strings, replace_strings): 
            if file_name in content_dict: # Perform the replacement in the content
                new_content_dict[file_name] = new_content_dict[file_name].replace(search_str, replace_str) 
        return new_content_dict

    ori_file_contents = {}
    mid_file_contents = {}
    all_file_names = []
    for k, v in file_contentes_dict.items():
        ori_file_contents[k] = v['old_content']
        mid_file_contents[k] = v['new_content']
        all_file_names.append(k)

    file_names, search_strings, replace_strings = pre_process_SearPlace(search_replace_text)
    new_file_contents = update_file_content(file_names, search_strings, replace_strings, mid_file_contents)

    stage2_edits ={file: [ori_file_contents.get(file, ""), new_file_contents.get(file, "")] for file in all_file_names}
    # print(stage2_edits.keys())

    file_level_diff = ""
    for fn, edit_list in stage2_edits.items():
        now_diff = difflib.unified_diff(edit_list[0].splitlines(), edit_list[-1].splitlines(), lineterm='')
        now_diff_content = f"diff --git a/{fn} b/{fn}\n"
        for diff_line in now_diff:
            if diff_line == "--- ":
                now_diff_content += diff_line + f"a/{fn}\n"
            elif diff_line == "+++ ":
                now_diff_content += diff_line + f"b/{fn}\n"
            else:
                now_diff_content += diff_line + "\n"

        file_level_diff += now_diff_content
    
    return file_level_diff


if __name__ == '__main__':
    # test_parse()

    # Example Usage
    code1 = """
class MyClass:
    def existing_method(self):
        pass
"""

    code2 = """
class MyClass:
    def existing_method(self):
        pass

    async def new_method(self):
        pass
"""

    print(is_just_new_function(code1, code2))  # Output: True
