import json
import csv
import os
import logging
from pydriller import Git, ModifiedFile
import subprocess
from typing import Dict, List, Optional
from pathlib import Path
import re
import argparse
from sklearn.feature_selection import (
    VarianceThreshold, SelectKBest, chi2, RFE, SelectFromModel,
    mutual_info_classif
)
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import warnings
import tree_sitter
from tree_sitter import Language, Parser
import tree_sitter_go
warnings.filterwarnings('ignore')

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        filename='file_collection.log',
        filemode='w'
    )

def get_base_directory() -> Path:
    base_dir = Path.home() / '.bug-collector'
    base_dir.mkdir(parents=True, exist_ok=True)
    projects_dir = base_dir / 'projects'
    projects_dir.mkdir(parents=True, exist_ok=True)
    return base_dir

def get_pr_details(owner: str, repo: str, pr_id: int):
    pr_path = f"/repos/{owner}/{repo}/pulls/{pr_id}"
    command = [
        "gh",
        "api",
        "-H",
        "Accept: application/vnd.github+json",
        "-H",
        "X-GitHub-Api-Version: 2022-11-28",
        pr_path
    ]

    output = subprocess.run(command, capture_output=True, text=True)
    return json.loads(output.stdout)

def get_commit_files(owner: str, repo: str, commit_sha: str):
    commit_path = f"/repos/{owner}/{repo}/commits/{commit_sha}"
    command = [
        "gh",
        "api",
        "-H",
        "Accept: application/vnd.github+json",
        "-H",
        "X-GitHub-Api-Version: 2022-11-28",
        commit_path
    ]

    output = subprocess.run(command, capture_output=True, text=True)
    return json.loads(output.stdout).get('files', [])

def is_go_file(file_path: str) -> bool:
    return file_path.endswith('.go') and not file_path.endswith('_test.go')

def get_buggy_commits_for_file(fix_commit_sha: str, filename: str, gr: Git) -> List[str]:
    """
    Get commits that last modified the lines that were fixed in the fix commit for a specific file
    """
    try:
        commit = gr.get_commit(fix_commit_sha)
        buggy_commits = gr.get_commits_last_modified_lines(commit)

        # Find commits related to any file containing this filename
        result = []
        for filepath, commits in buggy_commits.items():
            if filename in filepath:  # Dosya adı yol içinde var mı diye kontrol et
                if isinstance(commits, set):
                    result.append(list(commits)[0])  # Son commit'i al
                else:
                    result.append(commits[0])  # Son commit'i al

        return list(set(result))  # Tekrar eden commit'leri temizle
    except Exception as e:
        logging.error(f"Error getting buggy commits for {fix_commit_sha}, file {filename}: {str(e)}")
        return []

def count_imports(source_code: str) -> int:
    """Count number of imports in Go source code"""
    if not source_code:
        return 0

    import_count = 0
    # Find all import blocks
    import_blocks = source_code.split('import (')

    for block in import_blocks[1:]:  # Skip first element which is code before first import
        end_idx = block.find(')')
        if end_idx == -1:
            continue

        # Get text between ( and )
        imports = block[:end_idx].strip()
        # Count non-empty lines
        import_count += len([line for line in imports.split('\n') if line.strip()])

    return import_count

def count_go_constructs(source_code: str) -> dict:
    """Count various Go-specific constructs using AST analysis via tree-sitter."""
    if not source_code:
        return {
            'struct_count': 0,
            'interface_count': 0,
            'loop_count': 0,
            'error_handling_count': 0,
            'goroutine_count': 0,
            'channel_count': 0,
            'defer_count': 0,
            'context_usage_count': 0,
            'json_tag_count': 0,
            'variadic_function_count': 0,
            'pointer_receiver_count': 0
        }

    try:
        # Initialize tree-sitter parser for Go
        language = Language(tree_sitter_go.language())
        parser = Parser(language)

        # Parse the source code
        tree = parser.parse(bytes(source_code, 'utf8'))
        root_node = tree.root_node

        # Initialize counters
        counts = {
            'struct_count': 0,
            'interface_count': 0,
            'loop_count': 0,
            'error_handling_count': 0,
            'goroutine_count': 0,
            'channel_count': 0,
            'defer_count': 0,
            'context_usage_count': 0,
            'json_tag_count': 0,
            'variadic_function_count': 0,
            'pointer_receiver_count': 0
        }

        def traverse_tree(node):
            """Recursively traverse the AST and count constructs."""
            node_type = node.type

            # Count struct definitions
            if node_type == 'type_declaration':
                for child in node.children:
                    if child.type == 'type_spec':
                        for spec_child in child.children:
                            if spec_child.type == 'struct_type':
                                counts['struct_count'] += 1
                            elif spec_child.type == 'interface_type':
                                counts['interface_count'] += 1

            # Count loops - only count for_statement, not range_clause since it's a child
            elif node_type == 'for_statement':
                counts['loop_count'] += 1

            # Count defer statements
            elif node_type == 'defer_statement':
                counts['defer_count'] += 1

            # Count go statements (goroutines)
            elif node_type == 'go_statement':
                counts['goroutine_count'] += 1

            # Count channel operations - only count make(chan T) calls to avoid double counting
            elif node_type == 'call_expression':
                # Check for make(chan T) calls
                if node.child_count >= 2:
                    func_name = node.children[0]
                    if func_name.type == 'identifier':
                        func_text = source_code[func_name.start_byte:func_name.end_byte]
                        if func_text == 'make':
                            args = node.children[1]  # argument_list
                            if args and args.child_count >= 2:
                                first_arg = args.children[1]  # skip opening paren
                                if first_arg.type == 'channel_type':
                                    counts['channel_count'] += 1

            # Count error handling patterns (if err != nil)
            elif node_type == 'if_statement':
                condition = node.child_by_field_name('condition')
                if condition and 'err' in source_code[condition.start_byte:condition.end_byte]:
                    counts['error_handling_count'] += 1

            # Count function declarations with context.Context parameter
            elif node_type == 'function_declaration':
                params = node.child_by_field_name('parameters')
                if params:
                    param_text = source_code[params.start_byte:params.end_byte]
                    if 'context.Context' in param_text:
                        counts['context_usage_count'] += 1

                    # Check for variadic functions
                    if '...' in param_text:
                        counts['variadic_function_count'] += 1

            # Count method declarations with pointer receivers
            elif node_type == 'method_declaration':
                receiver = node.child_by_field_name('receiver')
                if receiver:
                    receiver_text = source_code[receiver.start_byte:receiver.end_byte]
                    if '*' in receiver_text:
                        counts['pointer_receiver_count'] += 1

                # Check for context.Context in method parameters
                params = node.child_by_field_name('parameters')
                if params:
                    param_text = source_code[params.start_byte:params.end_byte]
                    if 'context.Context' in param_text:
                        counts['context_usage_count'] += 1

                    # Check for variadic methods
                    if '...' in param_text:
                        counts['variadic_function_count'] += 1

            # Count interface method declarations (method_elem)
            elif node_type == 'method_elem':
                # Check for context.Context in method parameters
                for child in node.children:
                    if child.type == 'parameter_list':
                        param_text = source_code[child.start_byte:child.end_byte]
                        if 'context.Context' in param_text:
                            counts['context_usage_count'] += 1

                        # Check for variadic methods
                        if '...' in param_text:
                            counts['variadic_function_count'] += 1

            # Count JSON tags in struct fields - look for raw_string_literal containing json:
            elif node_type == 'field_declaration':
                for child in node.children:
                    if child.type == 'raw_string_literal':
                        tag_text = source_code[child.start_byte:child.end_byte]
                        if 'json:' in tag_text:
                            counts['json_tag_count'] += 1

            # Recursively traverse child nodes
            for child in node.children:
                traverse_tree(child)

        # Start traversal from root
        traverse_tree(root_node)

        return counts

    except Exception as e:
        logging.error(f"Error parsing Go code with tree-sitter: {str(e)}")
        # Fallback to basic counting if AST parsing fails
        return {
            'struct_count': source_code.count('type ') if 'struct' in source_code else 0,
            'interface_count': source_code.count('type ') if 'interface' in source_code else 0,
            'loop_count': source_code.count('for '),
            'error_handling_count': source_code.count('if err '),
            'goroutine_count': source_code.count('go '),
            'channel_count': source_code.count('chan '),
            'defer_count': source_code.count('defer '),
            'context_usage_count': source_code.count('context.Context'),
            'json_tag_count': source_code.count('json:'),
            'variadic_function_count': source_code.count('...'),
            'pointer_receiver_count': 0
        }

def calculate_method_metrics(methods) -> dict:
    """Calculate average method metrics"""
    if not methods:
        return {
            'avg_method_complexity': 0,  # Yeni
            'avg_methods_token_count': 0  # Yeni
        }

    total_metrics = {
        'method_complexity': 0,  # Yeni
        'methods_token_count': 0  # Yeni
    }

    for method in methods:
        total_metrics['method_complexity'] += method.complexity if method.complexity else 0  # Yeni
        total_metrics['methods_token_count'] += method.token_count if method.token_count else 0  # Yeni

    method_count = len(methods)
    return {
        'avg_method_complexity': total_metrics['method_complexity'] / method_count,  # Yeni
        'avg_methods_token_count': total_metrics['methods_token_count'] / method_count  # Yeni
    }

def process_project(project_dir: str, owner: str, repo: str):
    output_dir = f"file_data/{repo}"
    os.makedirs(output_dir, exist_ok=True)

    with open(f"{project_dir}/bug_prs.json", 'r') as f:
        prs = json.load(f)

    base_dir = get_base_directory()
    repo_path = str(base_dir / 'projects' / repo)
    gr = Git(repo_path)

    fieldnames = [
        'project', 'file_path', 'sha', 'nloc', 'complexity', 'token_count', 'method_count',
        'commit_count', 'authors_count', 'avg_method_param_count', 'import_count',
        'cyclo_per_loc', 'comment_ratio', 'struct_count', 'interface_count',
        'loop_count', 'error_handling_count', 'goroutine_count', 'channel_count',
        'defer_count', 'context_usage_count', 'json_tag_count', 'variadic_function_count',
        'pointer_receiver_count', 'avg_method_complexity', 'avg_methods_token_count'  # Yeni alanlar
    ]

    # İki ayrı dosya açalım - biri buggy diğeri non-bug metrikler için
    bug_file = f"{output_dir}/file_bug_metrics.csv"
    non_bug_file = f"{output_dir}/file_non_bug_metrics.csv"

    # Her iki dosyaya header yazalım
    for filename in [bug_file, non_bug_file]:
        with open(filename, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

    for pr in prs:
        try:
            pr_details = get_pr_details(owner, repo, pr['number'])
            if not pr_details.get('merge_commit_sha'):
                continue

            fix_commit = pr_details['merge_commit_sha']
            commit = gr.get_commit(fix_commit)

            # Sadece Go dosyalarını al
            modified_files = [f for f in commit.modified_files if is_go_file(f.filename)]

            for file in modified_files:
                # Non-bug versiyonu kaydet
                if file.nloc is not None:
                    # commit_count ve authors_count hesaplama
                    commits = gr.get_commits_modified_file(file.new_path)
                    try:
                        commit_idx = commits.index(fix_commit)
                        commit_count = len(commits[commit_idx:])
                        # Benzersiz yazar sayısını hesapla
                        authors = set()
                        for commit_sha in commits[commit_idx:]:
                            commit = gr.get_commit(commit_sha)
                            authors.add(commit.author.name)
                        authors_count = len(authors)
                    except ValueError:
                        commit_count = 1
                        authors_count = 1

                    # Metot parametre sayılarının ortalamasını hesapla
                    total_params = sum(len(method.parameters) for method in file.methods) if file.methods else 0
                    avg_params = total_params / len(file.methods) if file.methods else 0

                    # Import sayısını hesapla
                    import_count = count_imports(file.source_code)

                    # Go-specific metrics
                    go_metrics = count_go_constructs(file.source_code)
                    cyclo_per_loc = file.complexity / file.nloc if file.nloc else 0

                    # Yorum satırı sayısını // arayarak hesapla
                    comment_lines = sum(1 for line in file.source_code.splitlines()
                                     if line.strip().startswith('//'))
                    total_lines = sum(1 for line in file.source_code.splitlines()
                                    if line.strip())
                    comment_ratio = comment_lines / total_lines if total_lines > 0 else 0

                    # Yeni metrik hesaplamaları ekleniyor
                    method_metrics = calculate_method_metrics(file.methods)

                    with open(non_bug_file, 'a', newline='') as f:
                        writer = csv.DictWriter(f, fieldnames=fieldnames)
                        writer.writerow({
                            'project': repo,
                            'file_path': file.new_path,  # filename -> new_path
                            'sha': fix_commit,
                            'nloc': file.nloc,
                            'complexity': file.complexity if file.complexity else 0,
                            'token_count': file.token_count if file.token_count else 0,
                            'method_count': len(file.methods) if file.methods else 0,
                            'commit_count': commit_count,  # Yeni alan eklendi
                            'authors_count': authors_count,
                            'avg_method_param_count': avg_params,  # Yeni alan eklendi
                            'import_count': import_count,  # Yeni alan eklendi
                            'cyclo_per_loc': cyclo_per_loc,
                            'comment_ratio': comment_ratio,
                            **go_metrics,  # Go-specific metrics
                            **method_metrics  # Yeni metrikler eklendi
                        })

                # Bu dosyaya ait hatalı (buggy) versiyonları bul
                buggy_commits = get_buggy_commits_for_file(fix_commit, file.new_path, gr)  # filename -> new_path

                for buggy_commit_sha in buggy_commits:
                    try:
                        buggy_commit = gr.get_commit(buggy_commit_sha)
                        # Hatalı versiyondaki aynı dosyayı bul
                        for buggy_file in buggy_commit.modified_files:
                            if buggy_file.new_path == file.new_path and buggy_file.nloc is not None:  # filename -> new_path
                                # commit_count ve authors_count hesaplama
                                commits = gr.get_commits_modified_file(buggy_file.new_path)
                                try:
                                    commit_idx = commits.index(buggy_commit_sha)
                                    commit_count = len(commits[commit_idx:])
                                    # Benzersiz yazar sayısını hesapla
                                    authors = set()
                                    for commit_sha in commits[commit_idx:]:
                                        commit = gr.get_commit(commit_sha)
                                        authors.add(commit.author.name)
                                    authors_count = len(authors)
                                except ValueError:
                                    commit_count = 1
                                    authors_count = 1

                                # Metot parametre sayılarının ortalamasını hesapla
                                total_params = sum(len(method.parameters) for method in buggy_file.methods) if buggy_file.methods else 0
                                avg_params = total_params / len(buggy_file.methods) if buggy_file.methods else 0

                                # Import sayısını hesapla
                                import_count = count_imports(buggy_file.source_code)

                                # Go-specific metrics
                                go_metrics = count_go_constructs(buggy_file.source_code)
                                cyclo_per_loc = buggy_file.complexity / buggy_file.nloc if buggy_file.nloc else 0

                                # Yorum satırı sayısını // arayarak hesapla
                                comment_lines = sum(1 for line in buggy_file.source_code.splitlines()
                                                 if line.strip().startswith('//'))
                                total_lines = sum(1 for line in buggy_file.source_code.splitlines()
                                                if line.strip())
                                comment_ratio = comment_lines / total_lines if total_lines > 0 else 0

                                # Buggy dosya için metrik hesaplamaları
                                method_metrics = calculate_method_metrics(buggy_file.methods)

                                with open(bug_file, 'a', newline='') as f:
                                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                                    writer.writerow({
                                        'project': repo,
                                        'file_path': buggy_file.new_path,  # filename -> new_path
                                        'sha': buggy_commit_sha,
                                        'nloc': buggy_file.nloc,
                                        'complexity': buggy_file.complexity if buggy_file.complexity else 0,
                                        'token_count': buggy_file.token_count if buggy_file.token_count else 0,
                                        'method_count': len(buggy_file.methods) if buggy_file.methods else 0,
                                        'commit_count': commit_count,  # Yeni alan eklendi
                                        'authors_count': authors_count,
                                        'avg_method_param_count': avg_params,  # Yeni alan eklendi
                                        'import_count': import_count,  # Yeni alan eklendi
                                        'cyclo_per_loc': cyclo_per_loc,
                                        'comment_ratio': comment_ratio,
                                        **go_metrics,  # Go-specific metrics
                                        **method_metrics  # Yeni metrikler eklendi
                                    })
                                break
                    except Exception as e:
                        logging.error(f"Error processing buggy commit {buggy_commit_sha}: {str(e)}")

        except Exception as e:
            logging.error(f"Error processing PR {pr['number']}: {str(e)}")

def combine_all_data():
    """Combines all file_bug_metrics.csv and file_non_bug_metrics.csv files from different projects."""
    # Create the 'all' and duplicates directory
    all_dir = "file_data/all"
    os.makedirs(all_dir, exist_ok=True)

    # Initialize combined CSV files and duplicate tracking
    combined_bugs = []
    combined_non_bugs = []
    duplicates = []
    seen_files = set()  # Keep track of unique file+sha combinations

    # Get list of all project directories
    file_data_dir = Path('file_data')
    project_dirs = [d for d in file_data_dir.iterdir() if d.is_dir() and d.name != 'all']

    # Get headers from existing files
    headers = [
        'project', 'file_path', 'sha', 'nloc', 'complexity', 'token_count', 'method_count',
        'commit_count', 'authors_count', 'avg_method_param_count', 'import_count',
        'cyclo_per_loc', 'comment_ratio', 'struct_count', 'interface_count',
        'loop_count', 'error_handling_count', 'goroutine_count', 'channel_count',
        'defer_count', 'context_usage_count', 'json_tag_count', 'variadic_function_count',
        'pointer_receiver_count', 'avg_method_complexity', 'avg_methods_token_count'
    ]

    # Process each project
    for project_dir in project_dirs:
        project_name = project_dir.name

        # Process file_bug_metrics.csv
        bugs_path = project_dir / "file_bug_metrics.csv"
        if bugs_path.exists():
            with open(bugs_path, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if 'project' not in row:
                        row['project'] = project_name

                    # Create unique identifier for this file
                    file_id = f"{row['file_path']}:{row['sha']}"

                    if file_id in seen_files:
                        duplicates.append(row)
                    else:
                        seen_files.add(file_id)
                        combined_bugs.append(row)

        # Process file_non_bug_metrics.csv
        non_bugs_path = project_dir / "file_non_bug_metrics.csv"
        if non_bugs_path.exists():
            with open(non_bugs_path, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if 'project' not in row:
                        row['project'] = project_name

                    # Create unique identifier for this file
                    file_id = f"{row['file_path']}:{row['sha']}"

                    if file_id in seen_files:
                        duplicates.append(row)
                    else:
                        seen_files.add(file_id)
                        combined_non_bugs.append(row)

    # Write combined file_bug_metrics.csv
    with open(os.path.join(all_dir, "file_bug_metrics.csv"), 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(combined_bugs)

    # Write combined file_non_bug_metrics.csv
    with open(os.path.join(all_dir, "file_non_bug_metrics.csv"), 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(combined_non_bugs)

    # Write duplicates to a separate file
    if duplicates:
        with open(os.path.join(all_dir, "duplicates.csv"), 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            writer.writerows(duplicates)

    logging.info(f"Combined data created in {all_dir}")
    logging.info(f"Total bug files: {len(combined_bugs)}")
    logging.info(f"Total non-bug files: {len(combined_non_bugs)}")
    logging.info(f"Total duplicate entries: {len(duplicates)}")

def main():
    setup_logging()

    parser = argparse.ArgumentParser(description='Process and analyze Go files from projects')
    parser.add_argument('--combine', action='store_true',
                       help='Only combine existing project data into a single directory')

    args = parser.parse_args()

    if args.combine:
        combine_all_data()
        return

    projects = [
        {'owner': 'hashicorp', 'repo': 'consul'},
        {'owner': 'hashicorp', 'repo': 'terraform'},
        {'owner': 'hashicorp', 'repo': 'vault'},
        {'owner': 'hashicorp', 'repo': 'nomad'},
        {'owner': 'hashicorp', 'repo': 'packer'},
        {'owner': 'kubernetes', 'repo': 'kubernetes'},
        {'owner': 'docker', 'repo': 'compose'},
        {'owner': 'gin-gonic', 'repo': 'gin'},
        {'owner': 'grafana', 'repo': 'grafana'},
        {'owner': 'caddyserver', 'repo': 'caddy'},
        {'owner': 'traefik', 'repo': 'traefik'},
        {'owner': 'minio', 'repo': 'minio'},
        {'owner': 'rclone', 'repo': 'rclone'},
        {'owner': 'go-gitea', 'repo': 'gitea'},
        {'owner': 'gofiber', 'repo': 'fiber'},
        {'owner': 'influxdata', 'repo': 'influxdb'}
    ]

    os.makedirs("file_data", exist_ok=True)

    for project in projects:
        owner_repo = f"{project['owner']}/{project['repo']}"
        logging.info(f"Processing project: {owner_repo}")

        project_dir = f"data/{project['repo']}"
        if not os.path.exists(project_dir):
            logging.warning(f"Project directory not found: {project_dir}")
            continue

        try:
            process_project(project_dir, project['owner'], project['repo'])
            logging.info(f"Successfully processed {owner_repo}")
        except Exception as e:
            logging.error(f"Error processing project {project['repo']}: {str(e)}")

    logging.info("File collection process completed")

if __name__ == "__main__":
    main()
