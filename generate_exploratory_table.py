#!/usr/bin/env python3
"""
Generate Exploratory Results Table for Appendix

This script generates Table X: Exploratory project-level results
(not included in primary statistical comparisons).

For projects that don't meet the PRIMARY thresholds but meet EXPLORATORY thresholds,
we report their performance metrics for transparency, while clearly marking them
as exploratory.

Output:
- table_exploratory_results.tex (LaTeX booktabs)
- table_exploratory_results.md (Markdown)
- table_exploratory_results.csv (CSV data)
"""

import pandas as pd
import json
from pathlib import Path
from datetime import datetime

# Import adequacy filter
from adequacy_filter import (
    get_exploratory_projects, get_insufficient_projects,
    get_project_status, AdequacyStatus, load_adequacy_data
)

BASE_DIR = Path(__file__).resolve().parent
ACADEMIC_OUTPUTS_DIR = BASE_DIR / 'academic_outputs'


def get_results_dir(level):
    """Get the results directory for a specific level."""
    return BASE_DIR / f"results_{level}_level"


def collect_exploratory_results(cv_type='temporal', feature_set='full'):
    """
    Collect results for EXPLORATORY projects only.

    Returns DataFrame with best configuration metrics for each exploratory project.
    """
    rows = []

    for level in ['commit', 'file', 'method']:
        exploratory_projects = get_exploratory_projects(level)

        if not exploratory_projects:
            continue

        results_dir = get_results_dir(level)

        for project in exploratory_projects:
            feature_set_dir = results_dir / project / cv_type / feature_set

            if not feature_set_dir.exists():
                continue

            best_mcc = -999
            best_config = None
            best_metrics = None

            # Find best configuration for this project
            for resampling_dir in feature_set_dir.iterdir():
                if not resampling_dir.is_dir():
                    continue

                summary_file = resampling_dir / "analysis_summary.json"
                if not summary_file.exists():
                    continue

                try:
                    with open(summary_file, 'r') as f:
                        data = json.load(f)

                    resampling = resampling_dir.name
                    models = data.get('models', {})

                    for model_name, model_data in models.items():
                        holdout = model_data.get('holdout_metrics', {})
                        mcc = holdout.get('mcc')

                        if mcc is not None and mcc > best_mcc:
                            best_mcc = mcc
                            best_config = {'model': model_name, 'resampling': resampling}
                            best_metrics = holdout

                except Exception as e:
                    print(f"Error reading {summary_file}: {e}")

            if best_config and best_metrics:
                # Get adequacy info
                adequacy_df = load_adequacy_data()
                level_cap = level.capitalize()
                adequacy_row = adequacy_df[
                    (adequacy_df['Project'] == project) &
                    (adequacy_df['Level'] == level_cap)
                ]

                note = ""
                if not adequacy_row.empty:
                    note = adequacy_row.iloc[0].get('Note', '')

                rows.append({
                    'Project': project,
                    'Level': level_cap,
                    'Model': best_config['model'],
                    'Resampling': best_config['resampling'],
                    'MCC': best_metrics.get('mcc'),
                    'F1': best_metrics.get('f1_bug'),
                    'PR-AUC': best_metrics.get('pr_auc'),
                    'Precision': best_metrics.get('precision_bug'),
                    'Recall': best_metrics.get('recall_bug'),
                    'Reason': note if note else 'Below PRIMARY thresholds'
                })

    return pd.DataFrame(rows)


def generate_latex_table(df, output_path):
    """Generate LaTeX booktabs table for exploratory results."""

    latex_content = r"""\begin{table}[htbp]
\centering
\caption{Exploratory project-level results (not included in primary statistical comparisons).
These projects did not meet the minimum minority-count thresholds
(Train+Val buggy $\geq$ 20 AND Holdout buggy $\geq$ 10) required for primary analyses.
Results are reported for completeness but should be interpreted with caution due to
high variance induced by limited minority samples.}
\label{tab:exploratory_results}
\small
\begin{tabular}{@{}llllrrrrl@{}}
\toprule
Project & Level & Model & Resamp. & MCC & F1 & PR-AUC & Prec. & Reason \\
\midrule
"""

    for _, row in df.iterrows():
        project = row['Project'].replace('_', r'\_')
        model = row['Model'].replace('_', r'\_')
        resampling = row['Resampling'].replace('_', r'\_')
        if len(resampling) > 8:
            resampling = resampling[:7] + '.'

        mcc = f"{row['MCC']:.3f}" if pd.notna(row['MCC']) else '--'
        f1 = f"{row['F1']:.3f}" if pd.notna(row['F1']) else '--'
        pr_auc = f"{row['PR-AUC']:.3f}" if pd.notna(row['PR-AUC']) else '--'
        prec = f"{row['Precision']:.3f}" if pd.notna(row['Precision']) else '--'
        reason = row['Reason'].replace('_', r'\_').replace('<', r'$<$')

        latex_content += f"{project} & {row['Level']} & {model} & {resampling} & {mcc} & {f1} & {pr_auc} & {prec} & {reason} \\\\\n"

    latex_content += r"""\bottomrule
\end{tabular}
\vspace{2mm}
\begin{tablenotes}
\small
\item \textit{Note:} These results are exploratory and not used to support comparative claims.
Statistical tests (Friedman/Nemenyi, CD diagrams) exclude these projects.
\end{tablenotes}
\end{table}
"""

    with open(output_path, 'w') as f:
        f.write(latex_content)

    print(f"LaTeX table saved to: {output_path}")


def generate_markdown_table(df, output_path):
    """Generate Markdown table for exploratory results."""

    md_content = f"""# Appendix Table X: Exploratory Project-Level Results

**Note:** These projects did not meet the minimum minority-count thresholds required for
primary statistical analyses (Train+Val buggy ≥ 20 AND Holdout buggy ≥ 10). Results are
reported for completeness but are **not included in primary statistical comparisons**
(Friedman/Nemenyi tests, CD diagrams, model ranking).

## Thresholds (defined a priori, based only on class counts):
- **PRIMARY**: Train+Val buggy ≥ 20 AND Holdout buggy ≥ 10
- **EXPLORATORY**: Train+Val buggy ≥ 5 AND Holdout buggy ≥ 3

## Results

| Project | Level | Best Model | Best Resampling | MCC | F1 | PR-AUC | Precision | Recall | Reason |
|---------|-------|------------|-----------------|-----|----|----|-----------|--------|--------|
"""

    for _, row in df.iterrows():
        mcc = f"{row['MCC']:.3f}" if pd.notna(row['MCC']) else '--'
        f1 = f"{row['F1']:.3f}" if pd.notna(row['F1']) else '--'
        pr_auc = f"{row['PR-AUC']:.3f}" if pd.notna(row['PR-AUC']) else '--'
        prec = f"{row['Precision']:.3f}" if pd.notna(row['Precision']) else '--'
        recall = f"{row['Recall']:.3f}" if pd.notna(row['Recall']) else '--'

        md_content += f"| {row['Project']} | {row['Level']} | {row['Model']} | {row['Resampling']} | {mcc} | {f1} | {pr_auc} | {prec} | {recall} | {row['Reason']} |\n"

    md_content += f"""
## Interpretation Guidelines

1. **High variance**: With limited minority samples, metrics can vary significantly
   across random seeds and minor configuration changes.

2. **Not generalizable**: These results may not generalize to other projects or
   time periods due to the small sample sizes.

3. **For reference only**: Use these results as supplementary information, not as
   evidence for comparative claims.
"""

    with open(output_path, 'w') as f:
        f.write(md_content)

    print(f"Markdown table saved to: {output_path}")


def main():
    """Generate exploratory results tables."""
    print("=" * 60)
    print("Generating Exploratory Results Table (Appendix Table X)")
    print("=" * 60)

    # Ensure output directory exists
    ACADEMIC_OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    # Collect exploratory results
    print("\nCollecting exploratory project results...")
    df = collect_exploratory_results(cv_type='temporal', feature_set='full')

    if df.empty:
        print("No exploratory results found!")
        return

    print(f"Found {len(df)} exploratory (project, level) pairs")
    print(df[['Project', 'Level', 'MCC', 'F1', 'Reason']].to_string(index=False))

    # Save CSV
    csv_path = ACADEMIC_OUTPUTS_DIR / 'table_exploratory_results.csv'
    df.to_csv(csv_path, index=False)
    print(f"\nCSV saved to: {csv_path}")

    # Generate LaTeX table
    latex_path = ACADEMIC_OUTPUTS_DIR / 'table_exploratory_results.tex'
    generate_latex_table(df, latex_path)

    # Generate Markdown table
    md_path = ACADEMIC_OUTPUTS_DIR / 'table_exploratory_results.md'
    generate_markdown_table(df, md_path)

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == '__main__':
    main()
