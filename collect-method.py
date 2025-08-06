import json
import csv
import os
import logging
from pydriller import Git
import subprocess
from typing import List, Dict
from pathlib import Path
import re
import lizard
import argparse
import tree_sitter
from tree_sitter import Language, Parser
import tree_sitter_go


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        filename="method_collection.log",
        filemode="w",
    )


def get_base_directory() -> Path:
    base_dir = Path.home() / ".bug-collector"
    base_dir.mkdir(parents=True, exist_ok=True)
    projects_dir = base_dir / "projects"
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
        pr_path,
    ]

    output = subprocess.run(command, capture_output=True, text=True)
    return json.loads(output.stdout)


def is_go_file(file_path: str) -> bool:
    return file_path.endswith(".go") and not file_path.endswith("_test.go")


def get_buggy_commits_for_file(
    fix_commit_sha: str, filename: str, gr: Git
) -> List[str]:
    """
    Get commits that last modified the lines that were fixed in the fix commit for a specific file
    """
    try:
        commit = gr.get_commit(fix_commit_sha)
        buggy_commits = gr.get_commits_last_modified_lines(commit)

        result = []
        for filepath, commits in buggy_commits.items():
            if filename in filepath:
                if isinstance(commits, set):
                    result.append(list(commits)[0])
                else:
                    result.append(commits[0])

        return list(set(result))
    except Exception as e:
        logging.error(
            f"Error getting buggy commits for {fix_commit_sha}, file {filename}: {str(e)}"
        )
        return []


def count_go_constructs_in_method(method_source: str) -> dict:
    """Count Go-specific constructs in a method using AST analysis via tree-sitter"""
    if not method_source:
        return {
            "defer_count": 0,
            "channel_count": 0,
            "goroutine_count": 0,
            "error_handling_count": 0,
            "loop_count": 0,
        }

    try:
        # Initialize tree-sitter parser for Go
        language = Language(tree_sitter_go.language())
        parser = Parser(language)

        # Parse the method source code
        tree = parser.parse(bytes(method_source, 'utf8'))
        root_node = tree.root_node

        # Initialize counters
        counts = {
            "defer_count": 0,
            "channel_count": 0,
            "goroutine_count": 0,
            "error_handling_count": 0,
            "loop_count": 0,
        }

        def traverse_tree(node):
            """Recursively traverse the AST and count constructs."""
            node_type = node.type

            # Count loops - only count for_statement
            if node_type == 'for_statement':
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
                        func_text = method_source[func_name.start_byte:func_name.end_byte]
                        if func_text == 'make':
                            args = node.children[1]  # argument_list
                            if args and args.child_count >= 2:
                                first_arg = args.children[1]  # skip opening paren
                                if first_arg.type == 'channel_type':
                                    counts['channel_count'] += 1

            # Count error handling patterns (if err != nil)
            elif node_type == 'if_statement':
                condition = node.child_by_field_name('condition')
                if condition and 'err' in method_source[condition.start_byte:condition.end_byte]:
                    counts['error_handling_count'] += 1

            # Recursively traverse child nodes
            for child in node.children:
                traverse_tree(child)

        # Start traversal from root
        traverse_tree(root_node)

        return counts

    except Exception as e:
        logging.error(f"Error parsing Go method code with tree-sitter: {str(e)}")
        # Fallback to basic counting if AST parsing fails
        return {
            "defer_count": method_source.count('defer '),
            "channel_count": method_source.count('chan '),
            "goroutine_count": method_source.count('go '),
            "error_handling_count": len(re.findall(r"if\s+.*err\s*[!:=]", method_source)),
            "loop_count": len(re.findall(r"\bfor\b", method_source)),
        }


class Method:
    """Hold method information and metrics"""

    def __init__(self, name: str, start_line: int, end_line: int, source: str):
        self.name = name
        self.start_line = start_line
        self.end_line = end_line
        self.source = source

        # Calculate metrics using lizard
        analysis = lizard.analyze_file.analyze_source_code("temp.go", source)
        if analysis.function_list:
            func = analysis.function_list[0]  # Assuming single function in source
            self.complexity = func.cyclomatic_complexity
            self.nloc = func.nloc
            self.token_count = func.token_count
            self.parameter_count = len(func.parameters)
            self.nesting_depth = func.max_nesting_depth
        else:
            # Default values if lizard fails
            self.complexity = 0
            self.nloc = len(source.split("\n"))
            self.token_count = len(source.split())
            self.parameter_count = 0
            self.nesting_depth = 0

        # Calculate Go specific metrics
        go_metrics = count_go_constructs_in_method(source)
        for key, value in go_metrics.items():
            setattr(self, key, value)


def find_methods_in_file(file_content: str) -> List[Method]:
    """Find all methods in a Go file"""
    methods = []
    # Go method pattern
    method_pattern = r"func\s*\([^)]+\)\s+\w+\s*\([^{]*\{"

    for match in re.finditer(method_pattern, file_content):
        start_pos = match.start()
        # Find matching closing brace
        brace_count = 1
        pos = match.end()

        while brace_count > 0 and pos < len(file_content):
            if file_content[pos] == "{":
                brace_count += 1
            elif file_content[pos] == "}":
                brace_count -= 1
            pos += 1

        if brace_count == 0:
            # Extract method name
            name_match = re.search(
                r"func\s*\([^)]+\)\s+(\w+)", file_content[start_pos:pos]
            )
            if name_match:
                name = name_match.group(1)
                # Get line numbers
                start_line = file_content.count("\n", 0, start_pos) + 1
                end_line = file_content.count("\n", 0, pos) + 1
                # Get method source
                source = file_content[start_pos:pos]

                methods.append(Method(name, start_line, end_line, source))

    return methods


def process_project(project_dir: str, owner: str, repo: str, force: bool = False):
    output_dir = f"method_data/{repo}"
    os.makedirs(output_dir, exist_ok=True)

    # Check if bug_prs.json exists
    bug_prs_path = f"{project_dir}/bug_prs.json"
    if not os.path.exists(bug_prs_path):
        logging.warning(f"bug_prs.json not found in {project_dir}, skipping project {repo}")
        return # Skip project if bug_prs.json is missing

    with open(bug_prs_path, "r") as f:
        prs = json.load(f)

    base_dir = get_base_directory()
    repo_path = str(base_dir / "projects" / repo)
    gr = Git(repo_path)

    fieldnames = [
        "project",
        "file_path",
        "sha",
        "method_name",
        "cyclomatic_complexity",
        "nloc",
        "token_count",
        "parameter_count",
        "defer_count",
        "channel_count",
        "goroutine_count",
        "error_handling_count",
        "loop_count",
    ]

    bug_file = f"{output_dir}/method_bug_metrics.csv"
    non_bug_file = f"{output_dir}/method_non_bug_metrics.csv"

    if force:
        for f_path in [bug_file, non_bug_file]:
            if os.path.exists(f_path):
                os.remove(f_path)
                logging.info(f"Removed existing file due to --force: {f_path}")

    for filename in [bug_file, non_bug_file]:
        with open(filename, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

    for pr in prs:
        try:
            pr_details = get_pr_details(owner, repo, pr["number"])
            if not pr_details.get("merge_commit_sha"):
                continue

            fix_commit = pr_details["merge_commit_sha"]
            commit = gr.get_commit(fix_commit)

            modified_files = [
                f for f in commit.modified_files if is_go_file(f.filename)
            ]

            for file in modified_files:
                if not file.source_code or not file.changed_methods:
                    continue

                # Store changed methods from fix version
                changed_method_names = [method.name for method in file.changed_methods]

                # Save fix version metrics for changed methods
                with open(non_bug_file, "a", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    for method in file.changed_methods:
                        source = file.source_code.split("\n")[
                            method.start_line - 1 : method.end_line
                        ]
                        source = "\n".join(source)

                        go_metrics = count_go_constructs_in_method(source)
                        metrics = {
                            "project": repo,
                            "file_path": file.new_path,
                            "sha": fix_commit,
                            "method_name": method.name,
                            "cyclomatic_complexity": method.complexity
                            if method.complexity
                            else 0,
                            "nloc": method.nloc if method.nloc else 0,
                            "token_count": method.token_count
                            if method.token_count
                            else 0,
                            "parameter_count": len(method.parameters)
                            if method.parameters
                            else 0,
                            "defer_count": go_metrics["defer_count"],
                            "channel_count": go_metrics["channel_count"],
                            "goroutine_count": go_metrics["goroutine_count"],
                            "error_handling_count": go_metrics["error_handling_count"],
                            "loop_count": go_metrics["loop_count"],
                        }
                        writer.writerow(metrics)

                # Process buggy version - only look for methods that were changed in fix version
                buggy_commits = get_buggy_commits_for_file(
                    fix_commit, file.new_path, gr
                )
                for buggy_commit_sha in buggy_commits:
                    try:
                        buggy_commit = gr.get_commit(buggy_commit_sha)
                        for buggy_file in buggy_commit.modified_files:
                            if (
                                buggy_file.new_path == file.new_path
                                and buggy_file.source_code
                                and buggy_file.methods
                            ):
                                # Find matching methods in buggy version
                                buggy_methods = [
                                    m
                                    for m in buggy_file.methods
                                    if m.name in changed_method_names
                                ]

                                if buggy_methods:
                                    with open(bug_file, "a", newline="") as f:
                                        writer = csv.DictWriter(
                                            f, fieldnames=fieldnames
                                        )
                                        for method in buggy_methods:
                                            source = buggy_file.source_code.split("\n")[
                                                method.start_line - 1 : method.end_line
                                            ]
                                            source = "\n".join(source)

                                            go_metrics = count_go_constructs_in_method(
                                                source
                                            )
                                            metrics = {
                                                "project": repo,
                                                "file_path": buggy_file.new_path,
                                                "sha": buggy_commit_sha,
                                                "method_name": method.name,
                                                "cyclomatic_complexity": method.complexity
                                                if method.complexity
                                                else 0,
                                                "nloc": method.nloc
                                                if method.nloc
                                                else 0,
                                                "token_count": method.token_count
                                                if method.token_count
                                                else 0,
                                                "parameter_count": len(
                                                    method.parameters
                                                )
                                                if method.parameters
                                                else 0,
                                                "defer_count": go_metrics[
                                                    "defer_count"
                                                ],
                                                "channel_count": go_metrics[
                                                    "channel_count"
                                                ],
                                                "goroutine_count": go_metrics[
                                                    "goroutine_count"
                                                ],
                                                "error_handling_count": go_metrics[
                                                    "error_handling_count"
                                                ],
                                                "loop_count": go_metrics["loop_count"],
                                            }
                                            writer.writerow(metrics)
                                break
                    except Exception as e:
                        logging.error(
                            f"Error processing buggy commit {buggy_commit_sha}: {str(e)}"
                        )
                        continue

        except Exception as e:
            logging.error(f"Error processing PR {pr['number']}: {str(e)}")
            continue


def main():
    setup_logging()

    projects = [
        {'owner': 'hashicorp', 'repo': 'consul'},
        {'owner': 'hashicorp', 'repo': 'terraform'},
        {'owner': 'hashicorp', 'repo': 'vault'},
        {'owner': 'hashicorp', 'repo': 'nomad'},
        {'owner': 'hashicorp', 'repo': 'packer'},
        {'owner': 'moby', 'repo': 'moby'},
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

    os.makedirs("method_data", exist_ok=True)

    parser = argparse.ArgumentParser(description="Collect method-level bug data from GitHub repositories")
    parser.add_argument(
        "--combine",
        action="store_true",
        help="Combine all project method data into a single directory (method_data/all)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force reprocessing of all PRs for method data, ignoring existing CSV files.",
    )
    args = parser.parse_args()

    if args.combine:
        combine_all_method_data(projects)
        logging.info("Method data combination process completed.")
        return

    logging.info(
        f"Starting method-level data collection process{' (force mode)' if args.force else ''}"
    )

    for project in projects:
        owner_repo = f"{project['owner']}/{project['repo']}"
        logging.info(f"Processing project: {owner_repo}")

        project_dir = f"data/{project['repo']}"
        if not os.path.exists(project_dir):
            logging.warning(f"Project directory not found: {project_dir}")
            continue

        try:
            process_project(project_dir, project["owner"], project["repo"], args.force)
            logging.info(f"Successfully processed {owner_repo}")
        except Exception as e:
            logging.error(f"Error processing project {project['repo']}: {str(e)}")

    logging.info("Method collection process completed")


def combine_all_method_data(projects_list: List[Dict]):
    """Combines all method_bug_metrics.csv and method_non_bug_metrics.csv files
    from different projects into single files in method_data/all."""
    all_dir = "method_data/all"
    os.makedirs(all_dir, exist_ok=True)

    combined_bugs_data = []
    combined_non_bugs_data = []

    # Define fieldnames based on process_project function
    fieldnames = [
        "project",  # Added project column
        "file_path",
        "sha",
        "method_name",
        "cyclomatic_complexity",
        "nloc",
        "token_count",
        "parameter_count",
        "defer_count",
        "channel_count",
        "goroutine_count",
        "error_handling_count",
        "loop_count",
    ]

    for project_info in projects_list:
        repo_name = project_info['repo']
        project_method_data_dir = f"method_data/{repo_name}"

        # Process method_bug_metrics.csv
        bug_metrics_path = os.path.join(project_method_data_dir, "method_bug_metrics.csv")
        if os.path.exists(bug_metrics_path):
            with open(bug_metrics_path, "r", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    row["project"] = repo_name  # Add project identifier
                    combined_bugs_data.append(row)
        else:
            logging.warning(f"File not found: {bug_metrics_path}")

        # Process method_non_bug_metrics.csv
        non_bug_metrics_path = os.path.join(project_method_data_dir, "method_non_bug_metrics.csv")
        if os.path.exists(non_bug_metrics_path):
            with open(non_bug_metrics_path, "r", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    row["project"] = repo_name  # Add project identifier
                    combined_non_bugs_data.append(row)
        else:
            logging.warning(f"File not found: {non_bug_metrics_path}")

    # Write combined method_bug_metrics.csv
    combined_bug_output_path = os.path.join(all_dir, "method_bug_metrics.csv")
    with open(combined_bug_output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(combined_bugs_data)
    logging.info(f"Combined bug method metrics saved to {combined_bug_output_path} ({len(combined_bugs_data)} rows)")

    # Write combined method_non_bug_metrics.csv
    combined_non_bug_output_path = os.path.join(all_dir, "method_non_bug_metrics.csv")
    with open(combined_non_bug_output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(combined_non_bugs_data)
    logging.info(f"Combined non-bug method metrics saved to {combined_non_bug_output_path} ({len(combined_non_bugs_data)} rows)")


if __name__ == "__main__":
    main()
