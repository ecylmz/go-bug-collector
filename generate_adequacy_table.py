#!/usr/bin/env python3
"""
Generate Dataset Adequacy Table for Academic Publication

This script generates Table X: Dataset adequacy for primary statistical analyses.
The table shows which (project, level) pairs meet the a priori thresholds for
inclusion in primary statistical comparisons.

Thresholds (defined a priori, based ONLY on class counts, NOT on model performance):
- PRIMARY: Train+Val buggy >= 20 AND Holdout buggy >= 10
- EXPLORATORY: Train+Val buggy >= 5 AND Holdout buggy >= 3
- INSUFFICIENT: Below exploratory thresholds

Usage:
    python generate_adequacy_table.py [--output-dir OUTPUT_DIR] [--format FORMAT]

Options:
    --output-dir: Directory for output files (default: academic_outputs)
    --format: Output format - 'latex', 'markdown', 'both' (default: both)
"""

import argparse
import pandas as pd
from pathlib import Path
from datetime import datetime

# Import thresholds from centralized module for consistency
from adequacy_thresholds import (
    MIN_TRAINVAL_MINORITY_PRIMARY,
    MIN_HOLDOUT_MINORITY_PRIMARY,
    MIN_TRAINVAL_MINORITY_EXPLORATORY,
    MIN_HOLDOUT_MINORITY_EXPLORATORY
)

# Data file patterns
FILE_PATTERNS = {
    'commit': ('bugs.csv', 'non_bugs.csv'),
    'file': ('file_bug_metrics.csv', 'file_non_bug_metrics.csv'),
    'method': ('method_bug_metrics.csv', 'method_non_bug_metrics.csv')
}


def assess_quality(trainval_buggy: int, holdout_buggy: int) -> tuple:
    """
    Assess dataset quality based on minority class counts.

    Returns: (status, note)
    """
    if trainval_buggy >= MIN_TRAINVAL_MINORITY_PRIMARY and holdout_buggy >= MIN_HOLDOUT_MINORITY_PRIMARY:
        return "PRIMARY", ""
    elif trainval_buggy >= MIN_TRAINVAL_MINORITY_EXPLORATORY and holdout_buggy >= MIN_HOLDOUT_MINORITY_EXPLORATORY:
        notes = []
        if trainval_buggy < MIN_TRAINVAL_MINORITY_PRIMARY:
            notes.append(f"tv buggy < {MIN_TRAINVAL_MINORITY_PRIMARY}")
        if holdout_buggy < MIN_HOLDOUT_MINORITY_PRIMARY:
            notes.append(f"ho buggy < {MIN_HOLDOUT_MINORITY_PRIMARY}")
        return "EXPLORATORY", "; ".join(notes)
    else:
        notes = []
        if trainval_buggy < MIN_TRAINVAL_MINORITY_EXPLORATORY:
            notes.append(f"tv buggy < {MIN_TRAINVAL_MINORITY_EXPLORATORY}")
        if holdout_buggy < MIN_HOLDOUT_MINORITY_EXPLORATORY:
            notes.append(f"ho buggy < {MIN_HOLDOUT_MINORITY_EXPLORATORY}")
        return "INSUFFICIENT", "; ".join(notes)


def collect_adequacy_data() -> pd.DataFrame:
    """Collect dataset adequacy information for all projects and levels."""
    rows = []

    for level in ['commit', 'file', 'method']:
        data_dir = Path(f'{level}_data')
        if not data_dir.exists():
            continue

        bug_file, non_bug_file = FILE_PATTERNS[level]

        for project_dir in sorted(data_dir.iterdir()):
            if not project_dir.is_dir() or project_dir.name == 'combined':
                continue

            project = project_dir.name
            bugs_path = project_dir / bug_file
            non_bugs_path = project_dir / non_bug_file

            if not bugs_path.exists() or not non_bugs_path.exists():
                continue

            try:
                bugs_df = pd.read_csv(bugs_path)
                non_bugs_df = pd.read_csv(non_bugs_path)

                bugs_df['is_bug'] = 1
                non_bugs_df['is_bug'] = 0
                df = pd.concat([bugs_df, non_bugs_df], ignore_index=True)

                if 'commit_timestamp' not in df.columns:
                    continue

                df = df.sort_values('commit_timestamp')
                n_total = len(df)

                split_idx = int(n_total * 0.8)
                trainval = df.iloc[:split_idx]
                holdout = df.iloc[split_idx:]

                n_trainval = len(trainval)
                n_trainval_buggy = int(trainval['is_bug'].sum())
                n_holdout = len(holdout)
                n_holdout_buggy = int(holdout['is_bug'].sum())

                status, note = assess_quality(n_trainval_buggy, n_holdout_buggy)

                rows.append({
                    'Project': project,
                    'Level': level.capitalize(),
                    'Train+Val N': n_trainval,
                    'Train+Val Buggy': n_trainval_buggy,
                    'Holdout N': n_holdout,
                    'Holdout Buggy': n_holdout_buggy,
                    'Status': status,
                    'Note': note
                })

            except Exception as e:
                print(f"Error processing {project}/{level}: {e}")

    return pd.DataFrame(rows)


def generate_latex_table(df: pd.DataFrame, output_path: Path):
    """Generate LaTeX booktabs table."""

    # Sort for better presentation
    df_sorted = df.sort_values(['Level', 'Project'])

    latex_content = r"""\begin{table}[htbp]
\centering
\caption{Dataset adequacy for primary statistical analyses. Thresholds are defined
\textit{a priori} based only on buggy class counts: \textsc{Primary} requires
$\geq$20 buggy instances in training/validation and $\geq$10 in holdout;
\textsc{Exploratory} requires $\geq$5 and $\geq$3 respectively.}
\label{tab:dataset_adequacy}
\small
\begin{tabular}{@{}llrrrrll@{}}
\toprule
Project & Level & \multicolumn{2}{c}{Train+Val} & \multicolumn{2}{c}{Holdout} & Status & Note \\
\cmidrule(lr){3-4} \cmidrule(lr){5-6}
 &  & N & Buggy & N & Buggy &  &  \\
\midrule
"""

    current_level = None
    for _, row in df_sorted.iterrows():
        if current_level is not None and row['Level'] != current_level:
            latex_content += r"\midrule" + "\n"
        current_level = row['Level']

        # Escape underscores in project names
        project = row['Project'].replace('_', r'\_')

        # Format status with emphasis
        if row['Status'] == 'PRIMARY':
            status = r'\textsc{Primary}'
        elif row['Status'] == 'EXPLORATORY':
            status = r'\textit{Exploratory}'
        else:
            status = r'\textit{Insufficient}'

        note = row['Note'].replace('_', r'\_') if row['Note'] else '--'

        latex_content += f"{project} & {row['Level']} & {row['Train+Val N']:,} & {row['Train+Val Buggy']} & {row['Holdout N']:,} & {row['Holdout Buggy']} & {status} & {note} \\\\\n"

    latex_content += r"""\bottomrule
\end{tabular}
\end{table}
"""

    with open(output_path, 'w') as f:
        f.write(latex_content)

    print(f"LaTeX table saved to: {output_path}")


def generate_markdown_table(df: pd.DataFrame, output_path: Path):
    """Generate Markdown table."""

    df_sorted = df.sort_values(['Level', 'Project'])

    md_content = """# Table X: Dataset Adequacy for Primary Statistical Analyses

**Thresholds (defined a priori, based ONLY on class counts, NOT on model performance):**
- **PRIMARY**: Train+Val buggy ≥ 20 AND Holdout buggy ≥ 10
- **EXPLORATORY**: Train+Val buggy ≥ 5 AND Holdout buggy ≥ 3
- **INSUFFICIENT**: Below exploratory thresholds

| Project | Level | Train+Val N | Train+Val Buggy | Holdout N | Holdout Buggy | Status | Note |
|---------|-------|-------------|-----------------|-----------|---------------|--------|------|
"""

    for _, row in df_sorted.iterrows():
        note = row['Note'] if row['Note'] else '--'
        md_content += f"| {row['Project']} | {row['Level']} | {row['Train+Val N']:,} | {row['Train+Val Buggy']} | {row['Holdout N']:,} | {row['Holdout Buggy']} | **{row['Status']}** | {note} |\n"

    # Add summary
    summary = df.groupby('Status').size()
    level_summary = df.groupby(['Level', 'Status']).size().unstack(fill_value=0)

    md_content += f"""
## Summary

### Overall
- **PRIMARY**: {summary.get('PRIMARY', 0)} (project, level) pairs
- **EXPLORATORY**: {summary.get('EXPLORATORY', 0)} (project, level) pairs
- **INSUFFICIENT**: {summary.get('INSUFFICIENT', 0)} (project, level) pairs

### By Level
"""

    for level in ['Commit', 'File', 'Method']:
        if level in level_summary.index:
            row = level_summary.loc[level]
            total = df[df['Level'] == level]['Project'].nunique()
            primary = row.get('PRIMARY', 0)
            md_content += f"- **{level}**: {primary}/{total} projects PRIMARY\n"

    md_content += f"""
## Methodology Note

> To avoid over-interpreting high-variance estimates on very small datasets, we distinguish
> between **primary** and **exploratory** project-level results. **Primary** comparative
> analyses (e.g., Friedman/Nemenyi tests, critical difference diagrams, and model ranking)
> are conducted only for (project, level) pairs where the **training window (first 80%)
> contains at least 20 buggy instances** and the **temporal holdout (last 20%) contains
> at least 10 buggy instances**. All remaining (project, level) pairs are still reported
> for completeness as **exploratory** cases, but we refrain from drawing strong comparative
> conclusions from them due to the increased variance induced by extremely limited minority
> samples. Importantly, these thresholds are defined a priori and depend only on class counts,
> not on model performance.
"""

    with open(output_path, 'w') as f:
        f.write(md_content)

    print(f"Markdown table saved to: {output_path}")


def generate_summary_csv(df: pd.DataFrame, output_path: Path):
    """Generate CSV for further analysis."""
    df.to_csv(output_path, index=False)
    print(f"CSV data saved to: {output_path}")


def print_summary(df: pd.DataFrame):
    """Print summary to console."""
    print("\n" + "="*80)
    print("DATASET ADEQUACY SUMMARY")
    print("="*80)

    print(f"\nThresholds (a priori, depend ONLY on class counts):")
    print(f"  PRIMARY: tv_buggy >= {MIN_TRAINVAL_MINORITY_PRIMARY} AND ho_buggy >= {MIN_HOLDOUT_MINORITY_PRIMARY}")
    print(f"  EXPLORATORY: tv_buggy >= {MIN_TRAINVAL_MINORITY_EXPLORATORY} AND ho_buggy >= {MIN_HOLDOUT_MINORITY_EXPLORATORY}")

    for level in ['Commit', 'File', 'Method']:
        level_df = df[df['Level'] == level]
        total = len(level_df)
        primary = len(level_df[level_df['Status'] == 'PRIMARY'])
        exploratory = len(level_df[level_df['Status'] == 'EXPLORATORY'])
        insufficient = len(level_df[level_df['Status'] == 'INSUFFICIENT'])

        print(f"\n{level.upper()} LEVEL:")
        print(f"  PRIMARY: {primary}/{total} projects")
        print(f"  EXPLORATORY: {exploratory}/{total} projects")
        print(f"  INSUFFICIENT: {insufficient}/{total} projects")

        # List non-primary projects
        non_primary = level_df[level_df['Status'] != 'PRIMARY']
        if len(non_primary) > 0:
            print(f"  Non-primary projects:")
            for _, row in non_primary.iterrows():
                print(f"    - {row['Project']}: {row['Status']} ({row['Note']})")

    # For paper text
    print("\n" + "="*80)
    print("TEXT FOR PAPER (copy-paste ready):")
    print("="*80)

    commit_primary = len(df[(df['Level'] == 'Commit') & (df['Status'] == 'PRIMARY')])
    file_primary = len(df[(df['Level'] == 'File') & (df['Status'] == 'PRIMARY')])
    method_primary = len(df[(df['Level'] == 'Method') & (df['Status'] == 'PRIMARY')])
    total_projects = df['Project'].nunique()

    print(f"""
After applying these criteria, we retain {commit_primary}/{total_projects} projects
at the commit level, {file_primary}/{total_projects} at the file level, and
{method_primary}/{total_projects} at the method level for primary statistical
comparisons (Appendix Table X lists the included projects and counts).
""")


def main():
    parser = argparse.ArgumentParser(
        description='Generate dataset adequacy table for academic publication'
    )
    parser.add_argument('--output-dir', type=str, default='academic_outputs',
                        help='Output directory (default: academic_outputs)')
    parser.add_argument('--format', type=str, default='both',
                        choices=['latex', 'markdown', 'both'],
                        help='Output format (default: both)')

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    # Collect data
    print("Collecting dataset adequacy information...")
    df = collect_adequacy_data()

    if df.empty:
        print("Error: No data found. Make sure *_data directories exist.")
        return

    # Generate outputs
    generate_summary_csv(df, output_dir / 'dataset_adequacy.csv')

    if args.format in ['markdown', 'both']:
        generate_markdown_table(df, output_dir / 'table_dataset_adequacy.md')

    if args.format in ['latex', 'both']:
        generate_latex_table(df, output_dir / 'table_dataset_adequacy.tex')

    # Print summary
    print_summary(df)


if __name__ == '__main__':
    main()
