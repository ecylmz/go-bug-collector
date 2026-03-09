#!/usr/bin/env python3
"""
Generate Project Inclusion Matrix Table

This script generates a matrix showing which projects are included in
PRIMARY statistical analyses at each granularity level.

Symbols:
  ✓ (checkmark) - PRIMARY: included in main statistical comparisons
  △ (triangle)  - EXPLORATORY: reported in Appendix, not in comparisons
  ✗ (cross)     - INSUFFICIENT: excluded entirely

Output:
- table_project_inclusion_matrix.tex (LaTeX booktabs)
- table_project_inclusion_matrix.md (Markdown)
"""

import pandas as pd
from pathlib import Path
from datetime import datetime

from adequacy_filter import (
    load_adequacy_data, AdequacyStatus,
    get_primary_projects, get_exploratory_projects, get_insufficient_projects
)

BASE_DIR = Path(__file__).resolve().parent
ACADEMIC_OUTPUTS_DIR = BASE_DIR / 'academic_outputs'


def generate_inclusion_matrix():
    """Generate the project inclusion matrix DataFrame."""

    adequacy_df = load_adequacy_data()

    # Get unique projects
    projects = sorted(adequacy_df['Project'].unique())
    levels = ['Commit', 'File', 'Method']

    rows = []
    for project in projects:
        row = {'Project': project}
        for level in levels:
            status_row = adequacy_df[
                (adequacy_df['Project'] == project) &
                (adequacy_df['Level'] == level)
            ]
            if status_row.empty:
                row[level] = '?'
            else:
                status = status_row.iloc[0]['Status']
                if status == 'PRIMARY':
                    row[level] = '✓'
                    row[f'{level}_status'] = 'PRIMARY'
                elif status == 'EXPLORATORY':
                    row[level] = '△'
                    row[f'{level}_status'] = 'EXPLORATORY'
                else:
                    row[level] = '✗'
                    row[f'{level}_status'] = 'INSUFFICIENT'
        rows.append(row)

    return pd.DataFrame(rows)


def get_counts_summary():
    """Get counts of PRIMARY/EXPLORATORY/INSUFFICIENT per level."""
    summary = {}
    for level in ['commit', 'file', 'method']:
        summary[level] = {
            'PRIMARY': len(get_primary_projects(level)),
            'EXPLORATORY': len(get_exploratory_projects(level)),
            'INSUFFICIENT': len(get_insufficient_projects(level))
        }
    return summary


def generate_latex_table(df, output_path):
    """Generate LaTeX booktabs table."""

    counts = get_counts_summary()

    latex_content = r"""\begin{table}[htbp]
\centering
\caption{Project inclusion matrix for primary statistical analyses.
{\checkmark} indicates projects meeting PRIMARY thresholds (included in Friedman/Nemenyi tests and CD diagrams);
$\triangle$ indicates EXPLORATORY projects (reported in Appendix, not in primary comparisons);
$\times$ indicates INSUFFICIENT data (excluded).
Thresholds: PRIMARY requires Train+Val buggy $\geq$ 20 AND Holdout buggy $\geq$ 10.}
\label{tab:project_inclusion_matrix}
\begin{tabular}{@{}l ccc @{}}
\toprule
Project & Commit & File & Method \\
\midrule
"""

    # Symbol mapping for LaTeX
    symbol_map = {
        '✓': r'\checkmark',
        '△': r'$\triangle$',
        '✗': r'$\times$'
    }

    for _, row in df.iterrows():
        project = row['Project'].replace('_', r'\_')
        commit_sym = symbol_map.get(row['Commit'], row['Commit'])
        file_sym = symbol_map.get(row['File'], row['File'])
        method_sym = symbol_map.get(row['Method'], row['Method'])

        latex_content += f"{project} & {commit_sym} & {file_sym} & {method_sym} \\\\\n"

    latex_content += r"""\midrule
"""

    # Add totals row
    latex_content += f"\\textbf{{PRIMARY Total}} & {counts['commit']['PRIMARY']} & {counts['file']['PRIMARY']} & {counts['method']['PRIMARY']} \\\\\n"
    latex_content += f"\\textbf{{EXPLORATORY Total}} & {counts['commit']['EXPLORATORY']} & {counts['file']['EXPLORATORY']} & {counts['method']['EXPLORATORY']} \\\\\n"
    latex_content += f"\\textbf{{INSUFFICIENT Total}} & {counts['commit']['INSUFFICIENT']} & {counts['file']['INSUFFICIENT']} & {counts['method']['INSUFFICIENT']} \\\\\n"

    latex_content += r"""\bottomrule
\end{tabular}
\vspace{2mm}
\begin{tablenotes}
\small
\item \textit{Note:} Thresholds were defined a priori based solely on buggy class counts,
before examining any model performance metrics. This prevents data-dependent exclusions.
\end{tablenotes}
\end{table}
"""

    with open(output_path, 'w') as f:
        f.write(latex_content)

    print(f"LaTeX table saved to: {output_path}")


def generate_markdown_table(df, output_path):
    """Generate Markdown table."""

    counts = get_counts_summary()

    md_content = f"""# Project Inclusion Matrix for Primary Statistical Analyses

This table shows which projects are included in PRIMARY statistical analyses at each
granularity level based on their buggy class counts in the temporal 80/20 split.

## Legend
- **✓** = PRIMARY: Included in main statistical comparisons (Friedman, Nemenyi, CD diagrams)
- **△** = EXPLORATORY: Results reported in Appendix, not included in primary comparisons
- **✗** = INSUFFICIENT: Excluded entirely due to extremely limited minority samples

## Thresholds (defined a priori):
- **PRIMARY**: Train+Val buggy ≥ 20 AND Holdout buggy ≥ 10
- **EXPLORATORY**: Train+Val buggy ≥ 5 AND Holdout buggy ≥ 3
- **INSUFFICIENT**: Below exploratory thresholds

## Inclusion Matrix

| Project | Commit | File | Method |
|---------|:------:|:----:|:------:|
"""

    for _, row in df.iterrows():
        md_content += f"| {row['Project']} | {row['Commit']} | {row['File']} | {row['Method']} |\n"

    md_content += f"""| **---** | **---** | **---** | **---** |
| **PRIMARY Total** | **{counts['commit']['PRIMARY']}** | **{counts['file']['PRIMARY']}** | **{counts['method']['PRIMARY']}** |
| **EXPLORATORY Total** | **{counts['commit']['EXPLORATORY']}** | **{counts['file']['EXPLORATORY']}** | **{counts['method']['EXPLORATORY']}** |
| **INSUFFICIENT Total** | **{counts['commit']['INSUFFICIENT']}** | **{counts['file']['INSUFFICIENT']}** | **{counts['method']['INSUFFICIENT']}** |

## Summary

| Level | PRIMARY | EXPLORATORY | INSUFFICIENT |
|-------|---------|-------------|--------------|
| Commit | {counts['commit']['PRIMARY']} | {counts['commit']['EXPLORATORY']} | {counts['commit']['INSUFFICIENT']} |
| File | {counts['file']['PRIMARY']} | {counts['file']['EXPLORATORY']} | {counts['file']['INSUFFICIENT']} |
| Method | {counts['method']['PRIMARY']} | {counts['method']['EXPLORATORY']} | {counts['method']['INSUFFICIENT']} |

## Implications for Statistical Analysis

1. **Commit Level**: {counts['commit']['PRIMARY']}/16 projects in PRIMARY analyses
2. **File Level**: {counts['file']['PRIMARY']}/16 projects in PRIMARY analyses
3. **Method Level**: {counts['method']['PRIMARY']}/16 projects in PRIMARY analyses

Projects marked with △ (EXPLORATORY) have their individual results reported in
Appendix Table X but are not used to support statistical claims about model rankings
or configuration effectiveness.
"""

    with open(output_path, 'w') as f:
        f.write(md_content)

    print(f"Markdown table saved to: {output_path}")


def main():
    """Generate project inclusion matrix tables."""
    print("=" * 60)
    print("Generating Project Inclusion Matrix")
    print("=" * 60)

    # Ensure output directory exists
    ACADEMIC_OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    # Generate matrix
    print("\nBuilding inclusion matrix...")
    df = generate_inclusion_matrix()

    # Print summary
    counts = get_counts_summary()
    print("\nSummary by Level:")
    for level in ['commit', 'file', 'method']:
        print(f"  {level.capitalize()}: PRIMARY={counts[level]['PRIMARY']}, "
              f"EXPLORATORY={counts[level]['EXPLORATORY']}, "
              f"INSUFFICIENT={counts[level]['INSUFFICIENT']}")

    # Display matrix
    print("\nInclusion Matrix:")
    print(df[['Project', 'Commit', 'File', 'Method']].to_string(index=False))

    # Save CSV (with detailed status)
    csv_path = ACADEMIC_OUTPUTS_DIR / 'table_project_inclusion_matrix.csv'
    df.to_csv(csv_path, index=False)
    print(f"\nCSV saved to: {csv_path}")

    # Generate LaTeX table
    latex_path = ACADEMIC_OUTPUTS_DIR / 'table_project_inclusion_matrix.tex'
    generate_latex_table(df, latex_path)

    # Generate Markdown table
    md_path = ACADEMIC_OUTPUTS_DIR / 'table_project_inclusion_matrix.md'
    generate_markdown_table(df, md_path)

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == '__main__':
    main()
