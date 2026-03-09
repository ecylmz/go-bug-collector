#!/usr/bin/env python3
"""
Dataset Adequacy Filter Module

This module provides reusable functions to filter projects based on their
adequacy status (PRIMARY/EXPLORATORY/INSUFFICIENT) for statistical analyses.

Thresholds (defined a priori, based ONLY on class counts, NOT on model performance):
- PRIMARY: Train+Val buggy >= 20 AND Holdout buggy >= 10
- EXPLORATORY: Train+Val buggy >= 5 AND Holdout buggy >= 3
- INSUFFICIENT: Below exploratory thresholds

Usage:
    from adequacy_filter import get_primary_projects, get_project_status, AdequacyStatus

    # Get list of PRIMARY projects for a specific level
    primary_projects = get_primary_projects('commit')

    # Check if a specific project-level pair is PRIMARY
    status = get_project_status('kubernetes', 'file')
    if status == AdequacyStatus.PRIMARY:
        # Include in statistical analyses
        pass
"""

import pandas as pd
from pathlib import Path
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple
from functools import lru_cache

# Base directory
BASE_DIR = Path(__file__).resolve().parent
ACADEMIC_OUTPUTS_DIR = BASE_DIR / 'academic_outputs'
ADEQUACY_CSV = ACADEMIC_OUTPUTS_DIR / 'dataset_adequacy.csv'

# Import thresholds from centralized module
from adequacy_thresholds import (
    MIN_TRAINVAL_MINORITY_PRIMARY,
    MIN_HOLDOUT_MINORITY_PRIMARY,
    MIN_TRAINVAL_MINORITY_EXPLORATORY,
    MIN_HOLDOUT_MINORITY_EXPLORATORY
)


class AdequacyStatus(Enum):
    """Project-level adequacy status for statistical analyses."""
    PRIMARY = "PRIMARY"           # Included in main statistical comparisons
    EXPLORATORY = "EXPLORATORY"   # Reported in appendix, not in primary comparisons
    INSUFFICIENT = "INSUFFICIENT" # Insufficient data for reliable metrics


@lru_cache(maxsize=1)
def load_adequacy_data() -> pd.DataFrame:
    """
    Load the dataset adequacy CSV file.

    Returns:
        DataFrame with columns: Project, Level, Train+Val N, Train+Val Buggy,
        Holdout N, Holdout Buggy, Status, Note
    """
    if not ADEQUACY_CSV.exists():
        raise FileNotFoundError(
            f"Adequacy data file not found: {ADEQUACY_CSV}\n"
            "Run 'python generate_adequacy_table.py' first."
        )

    df = pd.read_csv(ADEQUACY_CSV)
    # Normalize level names to lowercase for consistent lookup
    df['Level_lower'] = df['Level'].str.lower()
    return df


def get_project_status(project: str, level: str) -> AdequacyStatus:
    """
    Get the adequacy status for a specific project-level pair.

    Args:
        project: Project name (e.g., 'kubernetes', 'gin')
        level: Analysis level ('commit', 'file', or 'method')

    Returns:
        AdequacyStatus enum value
    """
    df = load_adequacy_data()
    level_lower = level.lower()

    mask = (df['Project'] == project) & (df['Level_lower'] == level_lower)
    matched = df[mask]

    if matched.empty:
        # Project-level pair not found in adequacy data
        return AdequacyStatus.INSUFFICIENT

    status_str = matched.iloc[0]['Status']
    return AdequacyStatus(status_str)


def get_primary_projects(level: str) -> List[str]:
    """
    Get list of PRIMARY projects for a specific level.

    Args:
        level: Analysis level ('commit', 'file', or 'method')

    Returns:
        List of project names that are PRIMARY at this level
    """
    df = load_adequacy_data()
    level_lower = level.lower()

    mask = (df['Level_lower'] == level_lower) & (df['Status'] == 'PRIMARY')
    return df[mask]['Project'].tolist()


def get_exploratory_projects(level: str) -> List[str]:
    """
    Get list of EXPLORATORY projects for a specific level.

    Args:
        level: Analysis level ('commit', 'file', or 'method')

    Returns:
        List of project names that are EXPLORATORY at this level
    """
    df = load_adequacy_data()
    level_lower = level.lower()

    mask = (df['Level_lower'] == level_lower) & (df['Status'] == 'EXPLORATORY')
    return df[mask]['Project'].tolist()


def get_insufficient_projects(level: str) -> List[str]:
    """
    Get list of INSUFFICIENT projects for a specific level.

    Args:
        level: Analysis level ('commit', 'file', or 'method')

    Returns:
        List of project names that are INSUFFICIENT at this level
    """
    df = load_adequacy_data()
    level_lower = level.lower()

    mask = (df['Level_lower'] == level_lower) & (df['Status'] == 'INSUFFICIENT')
    return df[mask]['Project'].tolist()


def get_all_project_statuses() -> Dict[Tuple[str, str], AdequacyStatus]:
    """
    Get a dictionary of all project-level pairs and their statuses.

    Returns:
        Dict mapping (project, level) tuples to AdequacyStatus values
    """
    df = load_adequacy_data()
    result = {}

    for _, row in df.iterrows():
        key = (row['Project'], row['Level_lower'])
        result[key] = AdequacyStatus(row['Status'])

    return result


def filter_dataframe_by_status(
    df: pd.DataFrame,
    level: str,
    project_column: str = 'project',
    status: AdequacyStatus = AdequacyStatus.PRIMARY
) -> pd.DataFrame:
    """
    Filter a DataFrame to include only projects with the specified status.

    Args:
        df: DataFrame to filter
        level: Analysis level ('commit', 'file', or 'method')
        project_column: Name of the column containing project names
        status: AdequacyStatus to filter by (default: PRIMARY)

    Returns:
        Filtered DataFrame
    """
    if status == AdequacyStatus.PRIMARY:
        allowed_projects = set(get_primary_projects(level))
    elif status == AdequacyStatus.EXPLORATORY:
        allowed_projects = set(get_exploratory_projects(level))
    else:
        allowed_projects = set(get_insufficient_projects(level))

    # Handle case-insensitive matching
    df_copy = df.copy()
    df_copy['_project_lower'] = df_copy[project_column].str.lower()
    allowed_lower = {p.lower() for p in allowed_projects}

    filtered = df_copy[df_copy['_project_lower'].isin(allowed_lower)]
    filtered = filtered.drop(columns=['_project_lower'])

    return filtered


def get_adequacy_summary() -> Dict[str, Dict[str, int]]:
    """
    Get a summary of project counts by status and level.

    Returns:
        Dict with structure: {level: {status: count}}
    """
    df = load_adequacy_data()

    summary = {}
    for level in ['commit', 'file', 'method']:
        level_df = df[df['Level_lower'] == level]
        summary[level] = {
            'PRIMARY': len(level_df[level_df['Status'] == 'PRIMARY']),
            'EXPLORATORY': len(level_df[level_df['Status'] == 'EXPLORATORY']),
            'INSUFFICIENT': len(level_df[level_df['Status'] == 'INSUFFICIENT']),
            'TOTAL': len(level_df)
        }

    return summary


def get_project_inclusion_matrix() -> pd.DataFrame:
    """
    Generate a project inclusion matrix showing status at each level.

    Returns:
        DataFrame with projects as rows, levels as columns, status symbols as values
        ✓ = PRIMARY, △ = EXPLORATORY, ✗ = INSUFFICIENT
    """
    df = load_adequacy_data()

    # Get unique projects
    projects = sorted(df['Project'].unique())
    levels = ['Commit', 'File', 'Method']

    # Status symbols
    symbols = {
        'PRIMARY': '✓',
        'EXPLORATORY': '△',
        'INSUFFICIENT': '✗'
    }

    matrix_data = []
    for project in projects:
        row = {'Project': project}
        for level in levels:
            mask = (df['Project'] == project) & (df['Level'] == level)
            if mask.any():
                status = df[mask].iloc[0]['Status']
                row[level] = symbols.get(status, '?')
            else:
                row[level] = '-'
        matrix_data.append(row)

    return pd.DataFrame(matrix_data)


def print_adequacy_summary():
    """Print a human-readable summary of dataset adequacy."""
    summary = get_adequacy_summary()

    print("=" * 60)
    print("Dataset Adequacy Summary")
    print("=" * 60)
    print("\nThresholds (a priori, based on class counts only):")
    print(f"  PRIMARY:     Train+Val buggy ≥ {MIN_TRAINVAL_MINORITY_PRIMARY}, Holdout buggy ≥ {MIN_HOLDOUT_MINORITY_PRIMARY}")
    print(f"  EXPLORATORY: Train+Val buggy ≥ {MIN_TRAINVAL_MINORITY_EXPLORATORY}, Holdout buggy ≥ {MIN_HOLDOUT_MINORITY_EXPLORATORY}")
    print(f"  INSUFFICIENT: Below exploratory thresholds")

    print("\nProject counts by level and status:")
    print("-" * 60)
    print(f"{'Level':<10} {'PRIMARY':>10} {'EXPLORATORY':>12} {'INSUFFICIENT':>13} {'TOTAL':>8}")
    print("-" * 60)

    for level in ['commit', 'file', 'method']:
        s = summary[level]
        print(f"{level.capitalize():<10} {s['PRIMARY']:>10} {s['EXPLORATORY']:>12} {s['INSUFFICIENT']:>13} {s['TOTAL']:>8}")

    print("-" * 60)

    # Print project lists
    print("\nPRIMARY projects by level (included in statistical comparisons):")
    for level in ['commit', 'file', 'method']:
        projects = get_primary_projects(level)
        print(f"  {level.capitalize()} ({len(projects)}): {', '.join(sorted(projects))}")

    print("\nEXPLORATORY projects (reported in Appendix, not in primary comparisons):")
    for level in ['commit', 'file', 'method']:
        projects = get_exploratory_projects(level)
        if projects:
            print(f"  {level.capitalize()}: {', '.join(sorted(projects))}")

    print("\nINSUFFICIENT projects (insufficient data for reliable metrics):")
    for level in ['commit', 'file', 'method']:
        projects = get_insufficient_projects(level)
        if projects:
            print(f"  {level.capitalize()}: {', '.join(sorted(projects))}")

    print("=" * 60)


if __name__ == '__main__':
    # Print summary when run directly
    print_adequacy_summary()

    # Print inclusion matrix
    print("\nProject Inclusion Matrix:")
    print("(✓ = PRIMARY, △ = EXPLORATORY, ✗ = INSUFFICIENT)")
    print("-" * 40)
    matrix = get_project_inclusion_matrix()
    print(matrix.to_string(index=False))
