import argparse
from pydriller import Git
import subprocess
import json
import csv
import logging
from datetime import datetime
import os
from typing import Dict, List, Optional
from pathlib import Path

# https://accessibleai.dev/post/extracting-git-data-pydriller/

projects = [
    # {"owner": "hashicorp", "repo": "consul", "label": "type/bug"},
    # {"owner": "hashicorp", "repo": "terraform", "label": "bug"},
    # {"owner": "hashicorp", "repo": "vault", "label": "bug"},
    # {"owner": "hashicorp", "repo": "nomad", "label": "type/bug"},
    # {"owner": "hashicorp", "repo": "packer", "label": "bug"},
    {'owner': 'kubernetes', 'repo': 'kubernetes', 'label': 'kind/bug'},
    # {'owner': 'docker', 'repo': 'compose', 'label': 'kind/bug'},
    # {'owner': 'gin-gonic', 'repo': 'gin', 'label': 'bug'},
    # {'owner': 'grafana', 'repo': 'grafana', 'label': 'type/bug'},
    # {'owner': 'caddyserver', 'repo': 'caddy', 'label': 'bug :lady_beetle:'},
    # {'owner': 'traefik', 'repo': 'traefik', 'label': 'kind/bug/fix'},
    # {'owner': 'minio', 'repo': 'minio', 'label': 'bugfix'},
    # {'owner': 'rclone', 'repo': 'rclone', 'label': 'bug'},
    # {'owner': 'go-gitea', 'repo': 'gitea', 'label': 'type/bug'},
    # {'owner': 'gofiber', 'repo': 'fiber', 'label': '☢️ Bug'},
    # {'owner': 'influxdata', 'repo': 'influxdb', 'label': 'kind/bug'}
]


def setup_logging():
    # Clear the log file
    with open("bug_collection.log", "w") as f:
        f.write("")

    # Setup logging in append mode
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        filename="bug_collection.log",
        filemode="a",
    )


def bug_fix_prs(owner: str, repository: str, label: str) -> List[Dict]:
    command = [
        "gh",
        "api",
        "--header",
        "Accept: application/vnd.github+json",
        "--method",
        "GET",
        f"/repos/{owner}/{repository}/issues",
        "-f",
        "state=closed",
        "-f",
        f"labels={label}",
        "-f",
        "per_page=100",  # Her sayfada 100 sonuç al
        "-f",
        "sort=updated",  # Güncellenme tarihine göre sırala
        "-f",
        "direction=desc",  # En yeni en üstte
        "--paginate",
    ]

    output = subprocess.run(command, capture_output=True, text=True)
    if output.stderr:
        logging.error(f"GitHub API error: {output.stderr}")

    try:
        prs = json.loads(output.stdout)
        logging.info(f"Found {len(prs)} total issues with label {label}")

        bug_fix = []
        for pr in prs:
            if "pull_request" in pr:
                bug_fix.append(pr)

        logging.info(f"Filtered to {len(bug_fix)} pull requests")
        return bug_fix
    except json.JSONDecodeError as e:
        logging.error(f"JSON parse error: {str(e)}")
        logging.error(f"Raw output: {output.stdout}")
        return []


def get_buggy_commits(fix_commit_sha, gr):
    try:
        commit = gr.get_commit(fix_commit_sha)
        buggy_commits = gr.get_commits_last_modified_lines(commit)

        # Convert set to list of commit hashes
        result = []
        for file_path, commits in buggy_commits.items():
            if isinstance(commits, set):
                result.extend(list(commits))
            else:
                result.extend(commits)
        return list(set(result))  # Remove duplicates
    except Exception as e:
        logging.error(f"Error getting buggy commits for {fix_commit_sha}: {str(e)}")
        return []


def write_json_to_file(data, filename):
    """
    Writes JSON data to a file.

    Parameters:
        data (dict): JSON data to write to the file.
        filename (str): Name of the file to write the JSON data to.
    """
    with open(filename, "w") as file:
        json.dump(data, file, indent=4)


def load_existing_data(output_dir: str) -> tuple[List[Dict], set, set]:
    """Load existing data from JSON and CSV files."""
    existing_prs = []
    processed_fixes = set()
    processed_bugs = set()

    # Load existing PRs
    json_file = f"{output_dir}/bug_prs.json"
    if os.path.exists(json_file):
        with open(json_file, "r") as f:
            existing_prs = json.load(f)

    # Load processed commit hashes from CSVs
    for file_path, commit_set in [
        (f"{output_dir}/non_bugs.csv", processed_fixes),
        (f"{output_dir}/bugs.csv", processed_bugs),
    ]:
        if os.path.exists(file_path):
            with open(file_path, "r") as f:
                reader = csv.DictReader(f)
                commit_set.update(row["sha"] for row in reader)

    return existing_prs, processed_fixes, processed_bugs


def get_base_directory() -> Path:
    """Get the base directory for bug collector data"""
    base_dir = Path.home() / ".bug-collector"
    base_dir.mkdir(parents=True, exist_ok=True)
    projects_dir = base_dir / "projects"
    projects_dir.mkdir(parents=True, exist_ok=True)
    return base_dir


def collect(project: Dict, force: bool = False):
    owner, repo, label = project["owner"], project["repo"], project["label"]
    base_dir = get_base_directory()
    repo_path = str(base_dir / "projects" / repo)
    output_dir = f"data/{repo}"
    os.makedirs(output_dir, exist_ok=True)

    # In force mode, don't load existing data
    if force:
        existing_prs = []
        processed_fixes = set()
        processed_bugs = set()
        # Remove existing CSV files
        for file in ["non_bugs.csv", "bugs.csv"]:
            file_path = os.path.join(output_dir, file)
            if os.path.exists(file_path):
                os.remove(file_path)
    else:
        existing_prs, processed_fixes, processed_bugs = load_existing_data(output_dir)

    # Get current PRs
    current_prs = bug_fix_prs(owner, repo, label)

    # In force mode or if no existing PRs, process all PRs
    if force or not existing_prs:
        new_prs = current_prs
    else:
        existing_pr_numbers = {pr["number"] for pr in existing_prs}
        new_prs = [pr for pr in current_prs if pr["number"] not in existing_pr_numbers]

    if not new_prs:
        logging.info(f"No {'PRs' if force else 'new PRs'} found for {owner}/{repo}")
        return

    # Update PR data file with all PRs
    write_json_to_file(
        current_prs if force else existing_prs + new_prs, f"{output_dir}/bug_prs.json"
    )

    # Ensure repo exists with full history
    if not os.path.exists(repo_path):
        logging.info(f"Cloning {owner}/{repo} with full history...")
        subprocess.run(
            [
                "git",
                "clone",
                "--no-single-branch",  # Get all branches
                "--no-shallow-submodules",  # Get full history of submodules
                "--depth",
                "999999",  # Get deep history
                f"https://github.com/{owner}/{repo}.git",
                repo_path,
            ]
        )
        # Fetch all refs and history
        subprocess.run(["git", "fetch", "--unshallow"], cwd=repo_path)
    else:
        logging.info(f"Updating {owner}/{repo} repository...")
        subprocess.run(["git", "fetch", "--all"], cwd=repo_path)
        subprocess.run(["git", "pull", "--all"], cwd=repo_path)

    gr = Git(repo_path)
    logging.info(f"Processing {len(new_prs)} new PRs for {owner}/{repo}")

    # Process only new PRs
    process_commits(
        new_prs, gr, output_dir, processed_fixes, processed_bugs, owner, repo
    )


def extract_file_metrics(file):
    return {
        "change_type": file.change_type,
        "added_lines": file.added_lines,
        "deleted_lines": file.deleted_lines,
        "changed_methods_count": len(file.changed_methods),
        "nloc": file.nloc,
        "complexity": file.complexity,
        "token_count": file.token_count,
    }


# TODO
# aynı hash'e sahip varsa atla hesapla
# içerisinde go kodu olmayan komitleri de atla


def is_go_file(file_path):
    return file_path.endswith(".go") and not file_path.endswith("_test.go")


def extract_commit_metrics(sha: str, gr: Git) -> Optional[Dict]:
    commit = gr.get_commit(sha)
    if not commit:
        return None

    go_files = [f for f in commit.modified_files if is_go_file(f.filename)]
    if not go_files:
        return None

    # Calculate file-level metrics
    file_metrics = {
        "total_token_count": 0,
        "total_nloc": 0,
        "total_complexity": 0,
        "total_changed_method_count": 0,
    }

    for file in go_files:
        metrics = extract_file_metrics(file)
        file_metrics["total_token_count"] += (
            metrics["token_count"] if metrics["token_count"] is not None else 0
        )
        file_metrics["total_nloc"] += (
            metrics["nloc"] if metrics["nloc"] is not None else 0
        )
        file_metrics["total_complexity"] += (
            metrics["complexity"] if metrics["complexity"] is not None else 0
        )
        file_metrics["total_changed_method_count"] += metrics["changed_methods_count"]

    # Base metrics - removed date-related fields
    metrics = {
        "sha": sha,
        "commit_message": commit.msg,  # Add commit message
        "is_merge": commit.merge,
        "parents_count": len(commit.parents),
        "modified_files_count": len(go_files),
        "deletions": commit.deletions,
        "insertions": commit.insertions,
        "net_lines": commit.insertions - commit.deletions,
        "dmm_unit_size": commit.dmm_unit_size,
        "dmm_unit_complexity": commit.dmm_unit_complexity,
        "dmm_unit_interfacing": commit.dmm_unit_interfacing,
    }

    # Combine all metrics
    churn_metrics = calculate_churn_metrics(go_files)
    return {**metrics, **churn_metrics, **file_metrics}


def calculate_churn_metrics(files) -> Dict:
    if not files:
        return {"code_churn": 0, "max_file_churn": 0, "avg_file_churn": 0}

    try:
        total_churn = sum(file.added_lines + file.deleted_lines for file in files)
        max_churn = max((file.added_lines + file.deleted_lines) for file in files)
        avg_churn = total_churn / len(files)

        return {
            "code_churn": total_churn,
            "max_file_churn": max_churn,
            "avg_file_churn": avg_churn,
        }
    except Exception as e:
        logging.error(f"Error calculating churn metrics: {str(e)}")
        return {
            "code_churn": 0,
            "max_file_churn": 0,
            "avg_file_churn": 0,
        }


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

    # issue listesinde pr'ın sha bilgisine erişilemiyor. PR id'sinden tek tek pr detaylarını da çekmek gerekiyor.
    output = subprocess.run(command, capture_output=True, text=True)
    return json.loads(output.stdout)


def process_commits(
    prs: List[Dict],
    gr: Git,
    output_dir: str,
    processed_fixes: set,
    processed_bugs: set,
    owner: str,
    repo: str,
):
    headers = [
        "sha",
        "is_merge",
        "parents_count",
        "modified_files_count",
        "code_churn",
        "max_file_churn",
        "avg_file_churn",
        "deletions",
        "insertions",
        "net_lines",
        "dmm_unit_size",
        "dmm_unit_complexity",
        "dmm_unit_interfacing",
        "total_token_count",
        "total_nloc",
        "total_complexity",
        "total_changed_method_count",
        "commit_message",
    ]

    with (
        open(f"{output_dir}/non_bugs.csv", "a", newline="") as fix_file,
        open(f"{output_dir}/bugs.csv", "a", newline="") as bug_file,
    ):
        fix_writer = csv.DictWriter(fix_file, fieldnames=headers)
        bug_writer = csv.DictWriter(bug_file, fieldnames=headers)

        # Write headers only if files are empty
        if fix_file.tell() == 0:
            fix_writer.writeheader()
        if bug_file.tell() == 0:
            bug_writer.writeheader()

        for pr in prs:
            try:
                pr_details = get_pr_details(owner, repo, pr["number"])
                if not pr_details.get("merge_commit_sha"):
                    continue

                # Process fix commit if not already processed
                fix_sha = pr_details["merge_commit_sha"]
                if fix_sha not in processed_fixes:
                    fix_metrics = extract_commit_metrics(fix_sha, gr)
                    if fix_metrics:
                        filtered_metrics = {
                            k: fix_metrics[k] for k in headers if k in fix_metrics
                        }
                        fix_writer.writerow(filtered_metrics)
                        processed_fixes.add(fix_sha)

                # Process buggy commits if not already processed
                buggy_commits = get_buggy_commits(fix_sha, gr)
                for commit_hash in buggy_commits:
                    if commit_hash not in processed_bugs:
                        try:
                            bug_metrics = extract_commit_metrics(commit_hash, gr)
                            if bug_metrics:
                                filtered_metrics = {
                                    k: bug_metrics[k]
                                    for k in headers
                                    if k in bug_metrics
                                }
                                bug_writer.writerow(filtered_metrics)
                                processed_bugs.add(commit_hash)
                        except Exception as e:
                            logging.error(
                                f"Error processing buggy commit {commit_hash}: {str(e)}"
                            )

            except Exception as e:
                logging.error(f"Error processing PR {pr['number']}: {str(e)}")


def combine_all_data():
    """Combines all bugs.csv and non_bugs.csv files from different projects into single files."""
    # Create the 'all' directory
    all_dir = "data/all"
    os.makedirs(all_dir, exist_ok=True)

    # Initialize combined CSV files
    combined_bugs = []
    combined_non_bugs = []

    # Get headers from existing files
    headers = [
        "sha",
        "is_merge",
        "parents_count",
        "modified_files_count",
        "code_churn",
        "max_file_churn",
        "avg_file_churn",
        "deletions",
        "insertions",
        "net_lines",
        "dmm_unit_size",
        "dmm_unit_complexity",
        "dmm_unit_interfacing",
        "total_token_count",
        "total_nloc",
        "total_complexity",
        "total_changed_method_count",
        "commit_message",
    ]

    # Process each project
    for project in projects:
        project_dir = f"data/{project['repo']}"

        # Process bugs.csv
        bugs_path = os.path.join(project_dir, "bugs.csv")
        if os.path.exists(bugs_path):
            with open(bugs_path, "r") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # Add project identifier
                    row["project"] = project["repo"]
                    combined_bugs.append(row)

        # Process non_bugs.csv
        non_bugs_path = os.path.join(project_dir, "non_bugs.csv")
        if os.path.exists(non_bugs_path):
            with open(non_bugs_path, "r") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # Add project identifier
                    row["project"] = project["repo"]
                    combined_non_bugs.append(row)

    # Add 'project' to headers
    headers = ["project"] + headers

    # Write combined bugs.csv
    with open(os.path.join(all_dir, "bugs.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(combined_bugs)

    # Write combined non_bugs.csv
    with open(os.path.join(all_dir, "non_bugs.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(combined_non_bugs)

    logging.info(f"Combined data created in {all_dir}")
    logging.info(f"Total bugs: {len(combined_bugs)}")
    logging.info(f"Total non-bugs: {len(combined_non_bugs)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Collect bug fix data from GitHub repositories"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force reprocessing of all PRs, ignoring existing data",
    )
    parser.add_argument(
        "--combine",
        action="store_true",
        help="Combine all project data into a single directory",
    )
    args = parser.parse_args()

    setup_logging()
    logging.info(
        f"Starting bug collection process{' (force mode)' if args.force else ''}"
    )

    if args.combine:
        combine_all_data()
        exit()

    for project in projects:
        owner_repo = f"{project['owner']}/{project['repo']}"
        logging.info(f"Processing project: {owner_repo} with label: {project['label']}")
        try:
            collect(project, args.force)
        except Exception as e:
            logging.error(f"Error processing project {project['repo']}: {str(e)}")

    logging.info("Bug collection process completed")
