#!/usr/bin/env python3
"""
Script to add commit timestamps to all CSV files across all projects and levels.

This script automatically processes:
- commit_data level: bugs.csv and non_bugs.csv
- file_data level: file_bug_metrics.csv and file_non_bug_metrics.csv
- method_data level: method_bug_metrics.csv and method_non_bug_metrics.csv

For all projects found in the respective data directories.
"""

import argparse
import csv
import subprocess
import sys
from pathlib import Path
from typing import Optional, List, Tuple


def get_commit_timestamp(repo_path: Path, sha: str) -> Optional[int]:
    """
    Get the commit timestamp (Unix timestamp) for a given SHA.

    Args:
        repo_path: Path to the git repository
        sha: Commit SHA hash

    Returns:
        Unix timestamp as integer, or None if commit not found
    """
    try:
        result = subprocess.run(
            ['git', 'log', '-1', '--format=%ct', sha],
            cwd=repo_path,
            capture_output=True,
            text=True,
            check=True,
            timeout=10
        )

        timestamp_str = result.stdout.strip()
        if timestamp_str:
            return int(timestamp_str)
        return None
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, ValueError, Exception):
        return None


def add_timestamps_to_csv(csv_path: Path, repo_path: Path, dry_run: bool = False) -> Tuple[bool, int, int]:
    """
    Add commit_timestamp column to CSV file.

    Args:
        csv_path: Path to input CSV file
        repo_path: Path to git repository
        dry_run: If True, only report what would be done without making changes

    Returns:
        Tuple of (success, rows_processed, timestamps_found)
    """
    if not csv_path.exists():
        return False, 0, 0

    if not repo_path.exists() or not (repo_path / '.git').exists():
        print(f"  ⚠️  Warning: Git repository not found: {repo_path}", file=sys.stderr)
        return False, 0, 0

    # Read CSV file
    rows = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames

        if fieldnames is None:
            print(f"  ❌ Error: CSV file has no headers: {csv_path}", file=sys.stderr)
            return False, 0, 0

        # Check if commit_timestamp already exists
        if 'commit_timestamp' in fieldnames:
            # Check if all rows already have timestamps
            rows = list(reader)
            has_all_timestamps = all(
                row.get('commit_timestamp') and row.get('commit_timestamp').strip()
                for row in rows
            )
            if has_all_timestamps:
                print(f"  ✓ Already has timestamps ({len(rows)} rows)", end='')
                return True, len(rows), len(rows)
            # Remove existing commit_timestamp from fieldnames
            fieldnames = [f for f in fieldnames if f != 'commit_timestamp']
        else:
            rows = list(reader)

        # Add commit_timestamp to fieldnames
        new_fieldnames = list(fieldnames) + ['commit_timestamp']

    if dry_run:
        print(f"  [DRY RUN] Would process {len(rows)} commits")
        return True, len(rows), 0

    # Process each row to get timestamps
    total_rows = len(rows)

    for i, row in enumerate(rows, 1):
        sha = row.get('sha', '')
        if not sha:
            row['commit_timestamp'] = None
            continue

        # Get timestamp
        timestamp = get_commit_timestamp(repo_path, sha)
        row['commit_timestamp'] = timestamp

    # Write updated CSV
    with open(csv_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=new_fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    timestamps_found = sum(1 for row in rows if row.get('commit_timestamp') is not None)
    return True, total_rows, timestamps_found


def get_projects_for_level(base_dir: Path) -> List[str]:
    """
    Get list of projects for a given level directory.

    Args:
        base_dir: Base directory (commit_data, file_data, or method_data)

    Returns:
        List of project names (excluding 'combined' and 'combine')
    """
    if not base_dir.exists():
        return []

    projects = []
    for item in base_dir.iterdir():
        if item.is_dir() and item.name not in ['combined', 'combine']:
            projects.append(item.name)

    return sorted(projects)


def process_level(level: str, base_dir: Path, repo_base: Path, dry_run: bool = False):
    """
    Process all CSV files for a given level.

    Args:
        level: Level name ('commit', 'file', or 'method')
        base_dir: Base directory for this level
        repo_base: Base directory for git repositories
        dry_run: If True, only report what would be done
    """
    print(f"\n{'='*70}")
    print(f"Processing {level.upper()} level")
    print(f"{'='*70}")

    # Define file mappings for each level
    file_mappings = {
        'commit': [
            ('bugs.csv', 'bugs.csv'),
            ('non_bugs.csv', 'non_bugs.csv'),
        ],
        'file': [
            ('file_bug_metrics.csv', 'file_bug_metrics.csv'),
            ('file_non_bug_metrics.csv', 'file_non_bug_metrics.csv'),
        ],
        'method': [
            ('method_bug_metrics.csv', 'method_bug_metrics.csv'),
            ('method_non_bug_metrics.csv', 'method_non_bug_metrics.csv'),
        ],
    }

    if level not in file_mappings:
        print(f"❌ Unknown level: {level}")
        return

    projects = get_projects_for_level(base_dir)
    if not projects:
        print(f"  No projects found in {base_dir}")
        return

    print(f"Found {len(projects)} projects: {', '.join(projects)}")

    total_files = 0
    total_success = 0
    total_rows = 0
    total_timestamps = 0

    for project in projects:
        print(f"\n📁 Project: {project}")
        project_dir = base_dir / project
        repo_path = repo_base / project

        for csv_filename, _ in file_mappings[level]:
            csv_path = project_dir / csv_filename

            if not csv_path.exists():
                print(f"  ⏭️  Skipping {csv_filename} (not found)")
                continue

            total_files += 1
            print(f"  📄 Processing {csv_filename}...", end=' ', flush=True)

            success, rows, timestamps = add_timestamps_to_csv(
                csv_path, repo_path, dry_run=dry_run
            )

            if success:
                total_success += 1
                total_rows += rows
                total_timestamps += timestamps
                if not dry_run:
                    print(f"✓ ({rows} rows, {timestamps} timestamps)")
                else:
                    print(f"[DRY RUN]")
            else:
                print(f"❌ Failed")

    print(f"\n{'─'*70}")
    print(f"Summary for {level.upper()} level:")
    print(f"  Files processed: {total_success}/{total_files}")
    print(f"  Total rows: {total_rows}")
    print(f"  Total timestamps: {total_timestamps}")
    print(f"{'─'*70}")


def main():
    parser = argparse.ArgumentParser(
        description='Add commit timestamps to all CSV files across all projects and levels'
    )
    parser.add_argument(
        '--repo-base',
        type=Path,
        default=Path.home() / '.bug-collector' / 'projects',
        help='Base directory for git repositories (default: ~/.bug-collector/projects)'
    )
    parser.add_argument(
        '--data-base',
        type=Path,
        default=Path.cwd(),
        help='Base directory for data files (default: current directory)'
    )
    parser.add_argument(
        '--level',
        choices=['commit', 'file', 'method', 'all'],
        default='all',
        help='Which level to process (default: all)'
    )
    parser.add_argument(
        '--project',
        type=str,
        default=None,
        help='Process only a specific project (default: all projects)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be done without making changes'
    )

    args = parser.parse_args()

    if args.dry_run:
        print("🔍 DRY RUN MODE - No files will be modified")

    # Define level directories
    levels = {
        'commit': args.data_base / 'commit_data',
        'file': args.data_base / 'file_data',
        'method': args.data_base / 'method_data',
    }

    # Determine which levels to process
    if args.level == 'all':
        levels_to_process = list(levels.keys())
    else:
        levels_to_process = [args.level]

    # Process each level
    for level in levels_to_process:
        base_dir = levels[level]

        if args.project:
            # Process only specific project
            project_dir = base_dir / args.project
            if not project_dir.exists():
                print(f"❌ Project directory not found: {project_dir}")
                continue

            # Get file mappings
            file_mappings = {
                'commit': [
                    ('bugs.csv', 'bugs.csv'),
                    ('non_bugs.csv', 'non_bugs.csv'),
                ],
                'file': [
                    ('file_bug_metrics.csv', 'file_bug_metrics.csv'),
                    ('file_non_bug_metrics.csv', 'file_non_bug_metrics.csv'),
                ],
                'method': [
                    ('method_bug_metrics.csv', 'method_bug_metrics.csv'),
                    ('method_non_bug_metrics.csv', 'method_non_bug_metrics.csv'),
                ],
            }

            repo_path = args.repo_base / args.project

            print(f"\n📁 Processing project: {args.project} (level: {level})")
            for csv_filename, _ in file_mappings.get(level, []):
                csv_path = project_dir / csv_filename
                if csv_path.exists():
                    print(f"  📄 Processing {csv_filename}...", end=' ', flush=True)
                    success, rows, timestamps = add_timestamps_to_csv(
                        csv_path, repo_path, dry_run=args.dry_run
                    )
                    if success:
                        if not args.dry_run:
                            print(f"✓ ({rows} rows, {timestamps} timestamps)")
                        else:
                            print(f"[DRY RUN]")
                    else:
                        print(f"❌ Failed")
        else:
            # Process all projects for this level
            process_level(level, base_dir, args.repo_base, dry_run=args.dry_run)

    print(f"\n{'='*70}")
    print("✅ Processing complete!")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()










