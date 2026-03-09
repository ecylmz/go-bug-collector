#!/usr/bin/env python3
"""
Generate Reproducibility Table for Resampling Methods in GoBug.

This script produces a table reporting training-set sizes after resampling
and the number of synthetic samples generated (and removed) for representative scenarios.

Outputs:
- CSV file with resampling statistics
- LaTeX table (booktabs) for academic paper
"""

import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict

# Resampling imports
from imblearn.over_sampling import SMOTE, RandomOverSampler, ADASYN
from imblearn.under_sampling import RandomUnderSampler, TomekLinks
from imblearn.combine import SMOTETomek, SMOTEENN

# Project structure
BASE_DIR = Path(__file__).resolve().parent
COMMIT_DATA_DIR = BASE_DIR / 'commit_data'
FILE_DATA_DIR = BASE_DIR / 'file_data'
METHOD_DATA_DIR = BASE_DIR / 'method_data'
ACADEMIC_OUTPUTS_DIR = BASE_DIR / 'academic_outputs'

# All projects (excluding 'combined')
ALL_PROJECTS = [
    'caddy', 'compose', 'consul', 'fiber', 'gin', 'gitea', 'grafana',
    'influxdb', 'kubernetes', 'minio', 'nomad', 'packer', 'rclone',
    'terraform', 'traefik', 'vault'
]

ALL_LEVELS = ['commit', 'file', 'method']


def get_data_dir(level):
    """Get data directory for a given level."""
    if level == 'commit':
        return COMMIT_DATA_DIR
    elif level == 'file':
        return FILE_DATA_DIR
    elif level == 'method':
        return METHOD_DATA_DIR
    else:
        raise ValueError(f"Invalid level: {level}")


def load_project_data(project_name, level, sort_by_time=True):
    """Load and combine data for a specific project and level."""
    data_dir = get_data_dir(level)
    project_data_dir = data_dir / project_name

    if level == 'commit':
        bugs_file = 'bugs.csv'
        non_bugs_file = 'non_bugs.csv'
    elif level == 'file':
        bugs_file = 'file_bug_metrics.csv'
        non_bugs_file = 'file_non_bug_metrics.csv'
    elif level == 'method':
        bugs_file = 'method_bug_metrics.csv'
        non_bugs_file = 'method_non_bug_metrics.csv'
    else:
        return None

    bugs_path = project_data_dir / bugs_file
    non_bugs_path = project_data_dir / non_bugs_file

    if not (bugs_path.exists() and non_bugs_path.exists()):
        return None

    bugs_df = pd.read_csv(bugs_path)
    non_bugs_df = pd.read_csv(non_bugs_path)

    if len(bugs_df) == 0 and len(non_bugs_df) == 0:
        return None

    bugs_df['is_bug'] = 1
    non_bugs_df['is_bug'] = 0

    combined_df = pd.concat([bugs_df, non_bugs_df], ignore_index=True)

    if sort_by_time and 'commit_timestamp' in combined_df.columns:
        combined_df = combined_df.sort_values(
            by=['commit_timestamp', 'sha'] if 'sha' in combined_df.columns else ['commit_timestamp'],
            ascending=True
        ).reset_index(drop=True)

    return combined_df


def get_feature_columns(level):
    """Get feature columns for a given level."""
    if level == 'commit':
        return [
            'files_changed', 'lines_added', 'lines_removed', 'lines_modified',
            'entropy', 'author_experience', 'author_recent_commits',
            'hour_of_day', 'day_of_week', 'commit_message_length',
        ]
    elif level == 'file':
        return [
            'file_size', 'complexity', 'num_functions', 'num_classes',
            'lines_of_code', 'comment_lines', 'blank_lines',
            'avg_function_length', 'max_function_length',
            'imports_count', 'file_age_days', 'num_commits',
            'num_authors', 'lines_added', 'lines_removed',
        ]
    elif level == 'method':
        return [
            'method_lines', 'complexity', 'parameters_count',
            'local_variables', 'nested_depth', 'method_calls',
            'returns_count', 'loops_count', 'conditionals_count',
        ]
    return []


def prepare_features(df, level):
    """Prepare feature set based on the level."""
    feature_columns = get_feature_columns(level)

    # Filter to existing columns
    existing_cols = [col for col in feature_columns if col in df.columns]

    if not existing_cols:
        # Fallback: use all numeric columns except identifiers
        exclude_cols = {'is_bug', 'sha', 'commit_timestamp', 'file_path', 'method_name',
                       'project', 'commit_hash', 'author', 'message'}
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        existing_cols = [col for col in numeric_cols if col not in exclude_cols]

    if not existing_cols:
        return None, None

    df_filled = df.fillna({col: 0 for col in existing_cols})
    X = df_filled[existing_cols]
    y = df_filled['is_bug']
    return X, y


def get_temporal_train_val_split(df, X, y):
    """
    Get temporal 80/20 split based on commit order.
    Returns X_trainval, y_trainval (first 80% chronologically).
    """
    n_samples = len(df)
    split_idx = int(n_samples * 0.8)

    X_trainval = X.iloc[:split_idx].reset_index(drop=True)
    y_trainval = y.iloc[:split_idx].reset_index(drop=True)

    return X_trainval, y_trainval


def compute_scenario_stats():
    """
    Compute statistics for all (project, level) combinations.
    Returns a list of dicts with stats.
    """
    stats = []

    for project in ALL_PROJECTS:
        for level in ALL_LEVELS:
            df = load_project_data(project, level, sort_by_time=True)
            if df is None:
                continue

            X, y = prepare_features(df, level)
            if X is None or y is None or X.empty:
                continue

            X_trainval, y_trainval = get_temporal_train_val_split(df, X, y)

            n0 = int((y_trainval == 0).sum())  # class 0 (non-bug)
            n1 = int((y_trainval == 1).sum())  # class 1 (bug)
            n = n0 + n1

            if n == 0:
                continue

            # Identify majority/minority based on actual counts
            n_maj = max(n0, n1)
            n_min = min(n0, n1)
            p_min = n_min / n

            stats.append({
                'project': project,
                'level': level,
                'n0': n0,  # non-bug count
                'n1': n1,  # bug count
                'n_maj': n_maj,
                'n_min': n_min,
                'n': n,
                'p_min': p_min,
                'X_trainval': X_trainval,
                'y_trainval': y_trainval
            })

    return stats


def select_representative_scenarios(stats):
    """
    Select worst-case (min p_min) and large-scale (max n) scenarios for each level.
    Returns scenarios for commit, file, and method levels.
    """
    if not stats:
        return {}

    scenarios = {}

    for level in ['commit', 'file', 'method']:
        level_stats = [s for s in stats if s['level'] == level]
        if not level_stats:
            continue

        # Worst-case: minimum minority class ratio (excluding p_min=0)
        valid_stats = [s for s in level_stats if s['p_min'] > 0]
        if valid_stats:
            worst_case = min(valid_stats, key=lambda x: x['p_min'])
            scenarios[f'{level}_worst_case'] = worst_case

        # Large-scale: maximum number of samples
        large_scale = max(level_stats, key=lambda x: x['n'])
        scenarios[f'{level}_large_scale'] = large_scale

    return scenarios


def apply_resampling_and_compute_stats(X, y, method_name, random_state=42):
    """
    Apply resampling method and compute statistics.
    Returns dict with before/after counts and synthetic_added/removed.

    Class 0 = non-bug, Class 1 = bug
    """
    n_before = len(y)
    n0_before = int((y == 0).sum())  # non-bug
    n1_before = int((y == 1).sum())  # bug

    # Determine which class is oversampled (for over-samplers, minority is oversampled)
    # In our dataset: class 0 = non-bug, class 1 = bug (defective)
    minority_label = 1 if n1_before <= n0_before else 0
    oversampled_label = minority_label  # Over-samplers increase minority class

    result = {
        'method': method_name,
        'nonbug_before': n0_before,
        'bug_before': n1_before,
        'n_before': n_before,
        'nonbug_after': n0_before,
        'bug_after': n1_before,
        'n_after': n_before,
        'synthetic_added': 0,
        'removed': 0,
        'minority_label': minority_label,
        'oversampled_label': oversampled_label if method_name in ['RandomOverSampler', 'SMOTE', 'ADASYN', 'SMOTE+Tomek', 'SMOTE+ENN'] else None
    }

    if method_name == 'none':
        return result

    try:
        if method_name == 'RandomOverSampler':
            sampler = RandomOverSampler(random_state=random_state)
            X_res, y_res = sampler.fit_resample(X, y)
            n_after = len(y_res)
            result['nonbug_after'] = int((y_res == 0).sum())
            result['bug_after'] = int((y_res == 1).sum())
            result['n_after'] = n_after
            result['synthetic_added'] = n_after - n_before
            result['removed'] = 0

        elif method_name == 'SMOTE':
            sampler = SMOTE(random_state=random_state)
            X_res, y_res = sampler.fit_resample(X, y)
            n_after = len(y_res)
            result['nonbug_after'] = int((y_res == 0).sum())
            result['bug_after'] = int((y_res == 1).sum())
            result['n_after'] = n_after
            result['synthetic_added'] = n_after - n_before
            result['removed'] = 0

        elif method_name == 'ADASYN':
            sampler = ADASYN(random_state=random_state)
            X_res, y_res = sampler.fit_resample(X, y)
            n_after = len(y_res)
            result['nonbug_after'] = int((y_res == 0).sum())
            result['bug_after'] = int((y_res == 1).sum())
            result['n_after'] = n_after
            result['synthetic_added'] = n_after - n_before
            result['removed'] = 0

        elif method_name == 'RandomUnderSampler':
            sampler = RandomUnderSampler(random_state=random_state)
            X_res, y_res = sampler.fit_resample(X, y)
            n_after = len(y_res)
            result['nonbug_after'] = int((y_res == 0).sum())
            result['bug_after'] = int((y_res == 1).sum())
            result['n_after'] = n_after
            result['synthetic_added'] = 0
            result['removed'] = n_before - n_after

        elif method_name == 'TomekLinks':
            sampler = TomekLinks()
            X_res, y_res = sampler.fit_resample(X, y)
            n_after = len(y_res)
            result['nonbug_after'] = int((y_res == 0).sum())
            result['bug_after'] = int((y_res == 1).sum())
            result['n_after'] = n_after
            result['synthetic_added'] = 0
            result['removed'] = n_before - n_after

        elif method_name == 'SMOTE+Tomek':
            # Two-step process for accurate decomposition
            # Step 1: SMOTE
            smote = SMOTE(random_state=random_state)
            X_smote, y_smote = smote.fit_resample(X, y)
            n_smote = len(y_smote)
            synthetic_added = n_smote - n_before

            # Step 2: TomekLinks cleaning
            tomek = TomekLinks()
            X_res, y_res = tomek.fit_resample(X_smote, y_smote)
            n_after = len(y_res)
            removed = n_smote - n_after

            result['nonbug_after'] = int((y_res == 0).sum())
            result['bug_after'] = int((y_res == 1).sum())
            result['n_after'] = n_after
            result['synthetic_added'] = synthetic_added
            result['removed'] = removed

        elif method_name == 'SMOTE+ENN':
            # Two-step process for accurate decomposition
            # Step 1: SMOTE
            smote = SMOTE(random_state=random_state)
            X_smote, y_smote = smote.fit_resample(X, y)
            n_smote = len(y_smote)
            synthetic_added = n_smote - n_before

            # Step 2: ENN cleaning
            from imblearn.under_sampling import EditedNearestNeighbours
            enn = EditedNearestNeighbours()
            X_res, y_res = enn.fit_resample(X_smote, y_smote)
            n_after = len(y_res)
            removed = n_smote - n_after

            result['nonbug_after'] = int((y_res == 0).sum())
            result['bug_after'] = int((y_res == 1).sum())
            result['n_after'] = n_after
            result['synthetic_added'] = synthetic_added
            result['removed'] = removed

    except Exception as e:
        print(f"Warning: {method_name} failed: {e}")
        # Return original counts on failure
        pass

    return result


def generate_resampling_table():
    """Main function to generate the resampling reproducibility table."""

    print("=" * 70)
    print("Generating Resampling Reproducibility Table for GoBug")
    print("=" * 70)

    # Step 1: Compute statistics for all scenarios
    print("\n[Step 1] Computing statistics for all (project, level) combinations...")
    stats = compute_scenario_stats()
    print(f"  Found {len(stats)} valid scenarios")

    # Step 2: Select representative scenarios (for all levels)
    print("\n[Step 2] Selecting representative scenarios for each level...")
    scenarios_dict = select_representative_scenarios(stats)

    if not scenarios_dict:
        print("Error: Could not find representative scenarios")
        return

    # Print selected scenarios
    for scenario_key, scenario in scenarios_dict.items():
        scenario_type = 'Worst-case' if 'worst_case' in scenario_key else 'Large-scale'
        print(f"\n  {scenario_key} ({scenario_type}):")
        print(f"    Project: {scenario['project']}, Level: {scenario['level']}")
        print(f"    NonBug(class0)={scenario['n0']}, Bug(class1)={scenario['n1']}, n={scenario['n']}, p_min={scenario['p_min']:.4f}")

    # Step 3: Apply resampling methods to each scenario
    print("\n[Step 3] Applying resampling methods...")

    resampling_methods = [
        'none',
        'RandomOverSampler',
        'SMOTE',
        'ADASYN',
        'RandomUnderSampler',
        'TomekLinks',
        'SMOTE+Tomek',
        'SMOTE+ENN'
    ]

    results = []

    # Process scenarios in order: commit, file, method (worst_case then large_scale)
    scenario_order = [
        'commit_worst_case', 'commit_large_scale',
        'file_worst_case', 'file_large_scale',
        'method_worst_case', 'method_large_scale'
    ]

    for scenario_key in scenario_order:
        if scenario_key not in scenarios_dict:
            continue
        scenario = scenarios_dict[scenario_key]
        scenario_type = 'worst_case' if 'worst_case' in scenario_key else 'large_scale'

        print(f"\n  Processing {scenario_key}: {scenario['project']} ({scenario['level']} level)")

        X_trainval = scenario['X_trainval']
        y_trainval = scenario['y_trainval']

        for method in resampling_methods:
            print(f"    - {method}...", end=" ")
            stats_result = apply_resampling_and_compute_stats(X_trainval, y_trainval, method)
            stats_result['scenario_id'] = scenario_type
            stats_result['project'] = scenario['project']
            stats_result['level'] = scenario['level']
            results.append(stats_result)
            print(f"done (n_after={stats_result['n_after']})")

    # Step 4: Create DataFrame and save CSV
    print("\n[Step 4] Saving results...")

    df_results = pd.DataFrame(results)

    # Reorder columns
    columns_order = [
        'scenario_id', 'project', 'level', 'method',
        'nonbug_before', 'bug_before', 'nonbug_after', 'bug_after',
        'synthetic_added', 'removed', 'minority_label', 'oversampled_label'
    ]
    df_results = df_results[columns_order]

    # Save CSV
    csv_path = ACADEMIC_OUTPUTS_DIR / 'resampling_reproducibility.csv'
    df_results.to_csv(csv_path, index=False)
    print(f"  CSV saved to: {csv_path}")

    # Step 5: Generate LaTeX table
    latex_path = ACADEMIC_OUTPUTS_DIR / 'resampling_reproducibility.tex'
    generate_latex_table(df_results, latex_path, scenarios_dict)
    print(f"  LaTeX saved to: {latex_path}")

    # Also save as markdown
    md_path = ACADEMIC_OUTPUTS_DIR / 'resampling_reproducibility.md'
    generate_markdown_table(df_results, md_path, scenarios_dict)
    print(f"  Markdown saved to: {md_path}")

    print("\n" + "=" * 70)
    print("Done!")
    print("=" * 70)

    return df_results


def generate_latex_table(df, output_path, scenarios_dict):
    """Generate LaTeX table with booktabs."""

    latex_content = r"""\begin{table}[htbp]
\centering
\caption{Resampling Effects on Training Set Sizes. Resampling applied only to training (Train+Val); holdout test remained untouched. Worst-case scenarios have the minimum class ratio per level; large-scale scenarios have the maximum number of samples per level.}
\label{tab:resampling_reproducibility}
\footnotesize
\begin{tabular}{lllrrrrrr}
\toprule
\textbf{Level} & \textbf{Scenario} & \textbf{Method} & \textbf{NonBug$_{\text{before}}$} & \textbf{Bug$_{\text{before}}$} & \textbf{NonBug$_{\text{after}}$} & \textbf{Bug$_{\text{after}}$} & \textbf{+Synth} & \textbf{-Rem} \\
\midrule
"""

    current_level = None
    current_scenario = None

    for _, row in df.iterrows():
        level = row['level']
        scenario_id = row['scenario_id']
        project = row['project']

        # Level label (only show on first row of level)
        if level != current_level:
            if current_level is not None:
                latex_content += r"\midrule" + "\n"
            current_level = level
            level_label = level.title()
        else:
            level_label = ""

        # Scenario label (show project name)
        scenario_label = f"{scenario_id.replace('_', ' ').title()} ({project})"

        # Format numbers
        latex_content += f"{level_label} & {scenario_label} & {row['method']} & {row['nonbug_before']:,} & {row['bug_before']:,} & {row['nonbug_after']:,} & {row['bug_after']:,} & {row['synthetic_added']:,} & {row['removed']:,} \\\\\n"

    latex_content += r"""\bottomrule
\end{tabular}
\end{table}
"""

    with open(output_path, 'w') as f:
        f.write(latex_content)


def generate_markdown_table(df, output_path, scenarios_dict):
    """Generate Markdown table for documentation."""

    md_content = """# Resampling Reproducibility Table

**Note:** Resampling applied only to training (Train+Val); holdout test remained untouched.

## Selected Scenarios (Per Level)

"""

    # Group scenarios by level
    for level in ['commit', 'file', 'method']:
        md_content += f"### {level.title()} Level\n\n"

        # Worst-case
        wc_key = f'{level}_worst_case'
        if wc_key in scenarios_dict:
            wc = scenarios_dict[wc_key]
            md_content += f"""**Worst-case (Minimum Minority Ratio):**
- Project: {wc['project']}
- Non-bug (class 0): {wc['n0']:,}
- Bug (class 1): {wc['n1']:,}
- Total: {wc['n']:,}
- Minority Ratio (p_min): {wc['p_min']:.4f}

"""

        # Large-scale
        ls_key = f'{level}_large_scale'
        if ls_key in scenarios_dict:
            ls = scenarios_dict[ls_key]
            md_content += f"""**Large-scale (Maximum Samples):**
- Project: {ls['project']}
- Non-bug (class 0): {ls['n0']:,}
- Bug (class 1): {ls['n1']:,}
- Total: {ls['n']:,}
- Minority Ratio (p_min): {ls['p_min']:.4f}

"""

    md_content += """## Resampling Effects

| Level | Scenario | Project | Method | NonBug_before | Bug_before | NonBug_after | Bug_after | Synthetic Added | Removed | Minority Label | Oversampled Label |
|-------|----------|---------|--------|---------------|------------|--------------|-----------|-----------------|---------|----------------|-------------------|
"""

    for _, row in df.iterrows():
        minority_label = row.get('minority_label', '-')
        oversampled_label = row.get('oversampled_label', '-')
        oversampled_str = str(oversampled_label) if oversampled_label is not None else '-'
        md_content += f"| {row['level']} | {row['scenario_id']} | {row['project']} | {row['method']} | {row['nonbug_before']:,} | {row['bug_before']:,} | {row['nonbug_after']:,} | {row['bug_after']:,} | {row['synthetic_added']:,} | {row['removed']:,} | {minority_label} | {oversampled_str} |\n"

    md_content += """
## Column Descriptions

- **NonBug_before / Bug_before:** Class counts in Train+Val before resampling (class 0 = non-bug, class 1 = bug/defective)
- **NonBug_after / Bug_after:** Class counts after resampling
- **Synthetic Added:** Number of synthetic samples generated (for oversamplers)
- **Removed:** Number of samples removed (for undersamplers/cleaning methods)
- **Minority Label:** The minority class label in the dataset (0 or 1). Value 1 means bug/defective is minority.
- **Oversampled Label:** For oversampling methods, which class was synthetically increased (always the minority class)

### Resampling Methods

1. **none:** No resampling (baseline)
2. **RandomOverSampler:** Random duplication of the minority class samples (label indicated in Oversampled Label column)
3. **SMOTE:** Synthetic Minority Over-sampling Technique (generates synthetic minority samples by interpolation)
4. **ADASYN:** Adaptive Synthetic Sampling (focuses on harder-to-learn minority samples)
5. **RandomUnderSampler:** Random removal of majority class samples
6. **TomekLinks:** Removes majority class samples that form Tomek links
7. **SMOTE+Tomek:** SMOTE (minority oversampling) followed by Tomek links cleaning
8. **SMOTE+ENN:** SMOTE (minority oversampling) followed by Edited Nearest Neighbors cleaning
"""

    with open(output_path, 'w') as f:
        f.write(md_content)


if __name__ == '__main__':
    # Ensure output directory exists
    ACADEMIC_OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    # Generate the table
    generate_resampling_table()
