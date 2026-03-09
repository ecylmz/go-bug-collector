#!/usr/bin/env python3
"""
Script to add commit timestamps to CSV files containing commit SHAs.

This script reads a CSV file with commit SHAs and adds a 'commit_timestamp' column
containing the Unix timestamp of each commit's commit date.
"""

import argparse
import csv
import subprocess
import sys
from pathlib import Path
from typing import Optional


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
        # Use git log to get commit date as Unix timestamp
        # %ct = commit date as Unix timestamp
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
    except subprocess.CalledProcessError:
        # Commit not found or other git error
        return None
    except subprocess.TimeoutExpired:
        # Git command timed out
        return None
    except ValueError:
        # Invalid timestamp format
        return None
    except Exception as e:
        print(f"Error getting timestamp for {sha}: {e}", file=sys.stderr)
        return None


def add_timestamps_to_csv(csv_path: Path, repo_path: Path, output_path: Optional[Path] = None):
    """
    Add commit_timestamp column to CSV file.

    Args:
        csv_path: Path to input CSV file
        repo_path: Path to git repository
        output_path: Path to output CSV file (default: overwrite input file)
    """
    if not csv_path.exists():
        print(f"Error: CSV file not found: {csv_path}", file=sys.stderr)
        sys.exit(1)

    if not repo_path.exists() or not (repo_path / '.git').exists():
        print(f"Error: Git repository not found: {repo_path}", file=sys.stderr)
        sys.exit(1)

    if output_path is None:
        output_path = csv_path

    # Read CSV file
    rows = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames

        if fieldnames is None:
            print("Error: CSV file has no headers", file=sys.stderr)
            sys.exit(1)

        # Check if commit_timestamp already exists
        if 'commit_timestamp' in fieldnames:
            print("Warning: 'commit_timestamp' column already exists. Overwriting...", file=sys.stderr)
            # Remove existing commit_timestamp from fieldnames
            fieldnames = [f for f in fieldnames if f != 'commit_timestamp']

        # Add commit_timestamp to fieldnames
        new_fieldnames = list(fieldnames) + ['commit_timestamp']

        # Read all rows
        for row in reader:
            rows.append(row)

    # Process each row to get timestamps
    total_rows = len(rows)
    print(f"Processing {total_rows} commits...")

    for i, row in enumerate(rows, 1):
        sha = row.get('sha', '')
        if not sha:
            print(f"Warning: Row {i} has no SHA, skipping...", file=sys.stderr)
            row['commit_timestamp'] = None
            continue

        # Get timestamp
        timestamp = get_commit_timestamp(repo_path, sha)
        row['commit_timestamp'] = timestamp

        # Progress indicator
        if i % 100 == 0 or i == total_rows:
            print(f"Processed {i}/{total_rows} commits...", end='\r')

    print(f"\nCompleted processing {total_rows} commits.")

    # Write updated CSV
    with open(output_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=new_fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Updated CSV saved to: {output_path}")

    # Print statistics
    timestamps_found = sum(1 for row in rows if row.get('commit_timestamp') is not None)
    timestamps_missing = total_rows - timestamps_found
    print(f"Statistics: {timestamps_found} timestamps found, {timestamps_missing} missing")


def main():
    parser = argparse.ArgumentParser(
        description='Add commit timestamps to CSV file containing commit SHAs'
    )
    parser.add_argument(
        '--csv',
        type=Path,
        required=True,
        help='Path to input CSV file'
    )
    parser.add_argument(
        '--repo',
        type=Path,
        default=Path.home() / '.bug-collector' / 'projects' / 'caddy',
        help='Path to git repository (default: ~/.bug-collector/projects/caddy)'
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=None,
        help='Path to output CSV file (default: overwrite input file)'
    )

    args = parser.parse_args()

    add_timestamps_to_csv(args.csv, args.repo, args.output)


if __name__ == '__main__':
    main()










