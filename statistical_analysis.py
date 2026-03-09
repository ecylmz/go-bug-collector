#!/usr/bin/env python3
"""
Statistical Analysis for Bug Prediction Study
==============================================

This script performs rigorous statistical analysis following academic standards:
1. Project-level aggregation (n=16 independent observations)
2. Friedman test for multiple comparisons
3. Post-hoc tests (Nemenyi, Holm-Bonferroni corrected Wilcoxon)
4. Critical Difference (CD) diagrams
5. Effect size calculations (Cliff's delta)

Key principles:
- Each project is treated as an independent observation
- For each project-method pair, we use holdout score (single unbiased estimate)
- No fold-level aggregation to avoid pseudo-replication
- Only PRIMARY projects are included in statistical comparisons
"""

import json
import warnings
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import friedmanchisquare, wilcoxon, rankdata
import scikit_posthocs as sp
import argparse
import logging

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Configuration
BASE_DIR = Path(__file__).resolve().parent
ALL_PROJECTS = [
    "caddy", "compose", "consul", "fiber", "gin", "gitea", "grafana",
    "influxdb", "kubernetes", "minio", "nomad", "packer", "rclone",
    "terraform", "traefik", "vault"
]
ALL_LEVELS = ["commit", "file", "method"]

# Import adequacy filter module for CSV-based filtering
try:
    from adequacy_filter import (
        get_primary_projects, get_exploratory_projects, 
        get_project_status, AdequacyStatus, get_adequacy_summary
    )
    ADEQUACY_FILTER_AVAILABLE = True
except ImportError:
    ADEQUACY_FILTER_AVAILABLE = False
    logging.warning("adequacy_filter module not available. Using JSON-based filtering only.")
ALL_RESAMPLING = ["none", "smote", "random_under", "near_miss", "tomek", 
                  "random_over", "adasyn", "borderline", "smote_tomek", "smote_enn", "rose"]
ALL_MODELS = [
    'naive_bayes', 'xgboost', 'random_forest', 'logistic_regression',
    'catboost', 'lightgbm', 'gradient_boosting', 'decision_tree',
    'voting', 'mlp', 'stacking'
]


def get_results_dir(level):
    """Get the results directory for a specific level."""
    return BASE_DIR / f"results_{level}_level"


def get_projects_for_analysis(level, quality_filter='primary'):
    """
    Get list of projects to include in analysis based on quality filter.
    
    Uses adequacy_filter module (CSV-based) as primary source.
    Falls back to returning all projects if module not available.
    
    Parameters:
    -----------
    level : str
        Analysis level ('commit', 'file', 'method')
    quality_filter : str
        'primary', 'exploratory', or 'all'
    
    Returns:
    --------
    list of project names
    """
    if not ADEQUACY_FILTER_AVAILABLE:
        logging.warning("adequacy_filter not available, including all projects")
        return ALL_PROJECTS
    
    if quality_filter == 'primary':
        projects = get_primary_projects(level)
    elif quality_filter == 'exploratory':
        # Include both primary and exploratory
        projects = get_primary_projects(level) + get_exploratory_projects(level)
    else:
        # 'all' includes everything
        projects = ALL_PROJECTS
    
    logging.info(f"Projects for {level} analysis (filter={quality_filter}): {len(projects)} projects")
    return projects


def collect_all_results(level, cv_type='temporal', feature_set='full', 
                        quality_filter='primary', include_quality_info=False,
                        use_csv_filter=True):
    """
    Collect all results from analysis_summary.json files.
    
    Parameters:
    -----------
    level : str
        Analysis level ('commit', 'file', 'method')
    cv_type : str
        CV type ('temporal' or 'shuffle')
    feature_set : str
        Feature set ('full' or 'no_go_metrics')
    quality_filter : str or None
        - 'primary': Only include primary quality projects (for main statistical analyses)
        - 'exploratory': Include both primary and exploratory (exclude only 'insufficient')
        - 'all' or None: Include all projects regardless of quality
    include_quality_info : bool
        If True, include dataset_quality columns in output
    use_csv_filter : bool
        If True, use CSV-based adequacy filter; otherwise use JSON metadata
    
    Returns a DataFrame with columns:
    - project, resampling, model, metric (holdout_mcc, holdout_f1, cv_mcc, etc.)
    - dataset_quality (if include_quality_info=True)
    """
    results_dir = get_results_dir(level)
    all_data = []
    quality_summary = {'primary': [], 'exploratory': [], 'insufficient': [], 'unknown': []}
    
    # Get allowed projects based on CSV filter
    if use_csv_filter and ADEQUACY_FILTER_AVAILABLE and quality_filter:
        allowed_projects = set(get_projects_for_analysis(level, quality_filter))
        logging.info(f"CSV-based filter: {len(allowed_projects)} projects allowed for {level}/{quality_filter}")
    else:
        allowed_projects = None  # Will use JSON-based filtering
    
    for project in ALL_PROJECTS:
        # Apply CSV-based filter first (more reliable)
        if allowed_projects is not None and project not in allowed_projects:
            logging.debug(f"Skipping {project} (not in CSV filter for {level}/{quality_filter})")
            continue
            
        feature_set_dir = results_dir / project / cv_type / feature_set
        
        if not feature_set_dir.exists():
            logging.warning(f"Directory not found: {feature_set_dir}")
            continue
        
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
                
                # Get dataset quality info (for logging and optional include)
                dataset_quality_info = data.get('dataset_quality', {})
                quality_level = dataset_quality_info.get('quality_level', 'unknown')
                is_primary = dataset_quality_info.get('is_primary', False)
                trainval_minority = dataset_quality_info.get('trainval_minority_count', 0)
                holdout_minority = dataset_quality_info.get('holdout_minority_count', 0)
                
                # Track quality for summary
                if quality_level not in quality_summary:
                    quality_summary[quality_level] = []
                if project not in quality_summary.get(quality_level, []):
                    quality_summary[quality_level].append(project)
                
                # If CSV filter not used, apply JSON-based filter as fallback
                if allowed_projects is None:
                    if quality_filter == 'primary' and quality_level != 'primary':
                        continue
                    elif quality_filter == 'exploratory' and quality_level == 'insufficient':
                        continue
                
                for model_name, model_data in models.items():
                    cv_metrics = model_data.get('cv_metrics', {})
                    holdout_metrics = model_data.get('holdout_metrics', {})
                    
                    record = {
                        'project': project,
                        'resampling': resampling,
                        'model': model_name,
                        'cv_mcc': cv_metrics.get('mcc'),
                        'cv_mcc_std': cv_metrics.get('mcc_std'),
                        'cv_f1': cv_metrics.get('f1_bug'),
                        'cv_f1_std': cv_metrics.get('f1_bug_std'),
                        'cv_pr_auc': cv_metrics.get('pr_auc'),
                        'cv_pr_auc_std': cv_metrics.get('pr_auc_std'),
                        'cv_roc_auc': cv_metrics.get('roc_auc'),
                        'holdout_mcc': holdout_metrics.get('mcc'),
                        'holdout_f1': holdout_metrics.get('f1_bug'),
                        'holdout_pr_auc': holdout_metrics.get('pr_auc'),
                        'holdout_roc_auc': holdout_metrics.get('roc_auc'),
                        'holdout_precision': holdout_metrics.get('precision_bug'),
                        'holdout_recall': holdout_metrics.get('recall_bug'),
                    }
                    
                    if include_quality_info:
                        record.update({
                            'dataset_quality': quality_level,
                            'is_primary': is_primary,
                            'trainval_minority': trainval_minority,
                            'holdout_minority': holdout_minority,
                        })
                    
                    all_data.append(record)
                    
            except Exception as e:
                logging.error(f"Error reading {summary_file}: {e}")
    
    # Log quality summary
    logging.info(f"Dataset quality summary for {level}/{cv_type}/{feature_set}:")
    logging.info(f"  Primary projects: {len(quality_summary.get('primary', []))} - {quality_summary.get('primary', [])}")
    logging.info(f"  Exploratory projects: {len(quality_summary.get('exploratory', []))} - {quality_summary.get('exploratory', [])}")
    logging.info(f"  Insufficient projects: {len(quality_summary.get('insufficient', []))} - {quality_summary.get('insufficient', [])}")
    
    if quality_filter == 'primary':
        logging.info(f"  Using quality_filter='primary': Only {len(quality_summary.get('primary', []))} projects included in analysis")
    
    return pd.DataFrame(all_data)


def create_project_method_matrix(df, group_col, metric='holdout_mcc'):
    """
    Create a matrix where rows are projects and columns are methods (models/resampling).
    
    For each project-method pair, we take the BEST score across other configurations.
    This ensures fair comparison (best-case scenario for each method).
    
    Parameters:
    -----------
    df : DataFrame with all results
    group_col : 'model' or 'resampling' - which dimension to compare
    metric : which metric to use for comparison
    
    Returns:
    --------
    DataFrame with projects as rows and methods as columns
    """
    # For each project and group_col, get the best score
    if group_col == 'model':
        # Compare models: for each model, take best across all resampling methods
        pivot = df.groupby(['project', 'model'])[metric].max().unstack()
    else:
        # Compare resampling: for each resampling, take best across all models
        pivot = df.groupby(['project', 'resampling'])[metric].max().unstack()
    
    return pivot


def create_project_method_matrix_fixed_config(df, group_col, fixed_col, fixed_value, metric='holdout_mcc'):
    """
    Create a matrix with a fixed configuration.
    
    E.g., compare models with resampling fixed to 'smote'
    """
    filtered = df[df[fixed_col] == fixed_value]
    pivot = filtered.groupby(['project', group_col])[metric].first().unstack()
    return pivot


def friedman_test(data_matrix):
    """
    Perform Friedman test on a project × method matrix.
    
    Parameters:
    -----------
    data_matrix : DataFrame with projects as rows, methods as columns
    
    Returns:
    --------
    dict with statistic, p-value, and rankings
    """
    # Remove rows with NaN
    clean_matrix = data_matrix.dropna()
    
    if len(clean_matrix) < 3:
        return {'error': 'Not enough data points'}
    
    # Friedman test requires at least 2 methods and 3 observations
    values = [clean_matrix[col].values for col in clean_matrix.columns]
    
    try:
        stat, p_value = friedmanchisquare(*values)
    except Exception as e:
        return {'error': str(e)}
    
    # Calculate average ranks (higher is better for MCC/F1)
    # Rank each row (project), higher value = lower rank number = better
    ranks = clean_matrix.rank(axis=1, ascending=False)
    avg_ranks = ranks.mean()
    
    return {
        'statistic': stat,
        'p_value': p_value,
        'n_projects': len(clean_matrix),
        'n_methods': len(clean_matrix.columns),
        'avg_ranks': avg_ranks.to_dict(),
        'data_matrix': clean_matrix
    }


def nemenyi_posthoc(data_matrix, alpha=0.05):
    """
    Perform Nemenyi post-hoc test after Friedman.
    
    Returns p-value matrix for all pairwise comparisons.
    """
    clean_matrix = data_matrix.dropna()
    
    if len(clean_matrix) < 3:
        return None
    
    # scikit-posthocs expects each group as a separate column
    try:
        # Transpose so methods are rows for posthoc_nemenyi_friedman
        p_values = sp.posthoc_nemenyi_friedman(clean_matrix.values)
        p_values.index = clean_matrix.columns
        p_values.columns = clean_matrix.columns
        return p_values
    except Exception as e:
        logging.error(f"Nemenyi test error: {e}")
        return None


def holm_bonferroni_wilcoxon(data_matrix, control=None, alpha=0.05):
    """
    Perform Holm-Bonferroni corrected pairwise Wilcoxon tests.
    
    If control is specified, compare all methods against the control.
    Otherwise, do all pairwise comparisons.
    """
    clean_matrix = data_matrix.dropna()
    methods = list(clean_matrix.columns)
    n_methods = len(methods)
    
    results = []
    
    if control:
        # Compare all against control
        comparisons = [(control, m) for m in methods if m != control]
    else:
        # All pairwise
        comparisons = [(methods[i], methods[j]) 
                       for i in range(n_methods) 
                       for j in range(i+1, n_methods)]
    
    for m1, m2 in comparisons:
        try:
            stat, p = wilcoxon(clean_matrix[m1], clean_matrix[m2], alternative='two-sided')
            results.append({
                'method1': m1,
                'method2': m2,
                'statistic': stat,
                'p_value': p
            })
        except Exception as e:
            logging.warning(f"Wilcoxon failed for {m1} vs {m2}: {e}")
    
    if not results:
        return pd.DataFrame()
    
    results_df = pd.DataFrame(results)
    
    # Holm-Bonferroni correction
    results_df = results_df.sort_values('p_value')
    n_comparisons = len(results_df)
    
    adjusted_p = []
    for i, row in enumerate(results_df.itertuples()):
        # Holm correction: multiply by (n - rank + 1)
        adj_p = min(row.p_value * (n_comparisons - i), 1.0)
        adjusted_p.append(adj_p)
    
    results_df['p_adjusted'] = adjusted_p
    results_df['significant'] = results_df['p_adjusted'] < alpha
    
    return results_df


def cliffs_delta(x, y):
    """
    Calculate Cliff's delta effect size.
    
    Interpretation:
    |d| < 0.147: negligible
    |d| < 0.33: small
    |d| < 0.474: medium
    |d| >= 0.474: large
    """
    n1, n2 = len(x), len(y)
    
    # Count dominance
    more = sum(1 for xi in x for yi in y if xi > yi)
    less = sum(1 for xi in x for yi in y if xi < yi)
    
    delta = (more - less) / (n1 * n2)
    
    # Effect size interpretation
    abs_d = abs(delta)
    if abs_d < 0.147:
        interpretation = 'negligible'
    elif abs_d < 0.33:
        interpretation = 'small'
    elif abs_d < 0.474:
        interpretation = 'medium'
    else:
        interpretation = 'large'
    
    return delta, interpretation


def calculate_effect_sizes(data_matrix):
    """Calculate Cliff's delta for all pairs of methods."""
    clean_matrix = data_matrix.dropna()
    methods = list(clean_matrix.columns)
    n_methods = len(methods)
    
    results = []
    for i in range(n_methods):
        for j in range(i+1, n_methods):
            m1, m2 = methods[i], methods[j]
            delta, interp = cliffs_delta(
                clean_matrix[m1].values, 
                clean_matrix[m2].values
            )
            results.append({
                'method1': m1,
                'method2': m2,
                'cliffs_delta': delta,
                'effect_size': interp,
                'favors': m1 if delta > 0 else m2
            })
    
    return pd.DataFrame(results)


def critical_difference_diagram(avg_ranks, n_projects, n_methods, alpha=0.05, 
                                  title="Critical Difference Diagram", output_path=None):
    """
    Draw a Critical Difference (CD) diagram.
    
    Methods that are not significantly different are connected with a bar.
    """
    # Calculate critical difference (Nemenyi)
    # CD = q_alpha * sqrt(k(k+1)/(6n))
    # q_alpha values for different k at alpha=0.05
    q_alpha_table = {
        2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728,
        6: 2.850, 7: 2.949, 8: 3.031, 9: 3.102,
        10: 3.164, 11: 3.219, 12: 3.268
    }
    
    k = n_methods
    n = n_projects
    
    if k not in q_alpha_table:
        # Approximate using k=11 or interpolation
        q_alpha = 3.219 if k <= 11 else 3.3
    else:
        q_alpha = q_alpha_table[k]
    
    cd = q_alpha * np.sqrt(k * (k + 1) / (6 * n))
    
    # Sort methods by average rank
    sorted_ranks = sorted(avg_ranks.items(), key=lambda x: x[1])
    methods = [m for m, _ in sorted_ranks]
    ranks = [r for _, r in sorted_ranks]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, max(4, len(methods) * 0.4)))
    
    # Draw axis
    ax.set_xlim(0.5, k + 0.5)
    ax.set_ylim(0, 1)
    
    # Draw rank axis at top
    ax.axhline(y=0.9, color='black', linewidth=1)
    for i in range(1, k + 1):
        ax.axvline(x=i, ymin=0.88, ymax=0.92, color='black', linewidth=1)
        ax.text(i, 0.95, str(i), ha='center', va='bottom', fontsize=10)
    
    # Draw CD bar
    cd_start = 1
    ax.plot([cd_start, cd_start + cd], [0.85, 0.85], 'k-', linewidth=2)
    ax.text((cd_start + cd_start + cd) / 2, 0.82, f'CD = {cd:.3f}', ha='center', va='top', fontsize=9)
    
    # Position methods
    y_positions = np.linspace(0.7, 0.1, len(methods))
    
    for i, (method, rank) in enumerate(sorted_ranks):
        y = y_positions[i]
        
        # Draw line from method name to rank position
        ax.plot([rank, rank], [0.88, y + 0.02], 'k-', linewidth=0.5)
        ax.scatter([rank], [y + 0.02], color='black', s=20, zorder=5)
        
        # Method name
        if rank <= (k + 1) / 2:
            ax.text(rank - 0.1, y, f'{method} ({rank:.2f})', ha='right', va='center', fontsize=9)
        else:
            ax.text(rank + 0.1, y, f'{method} ({rank:.2f})', ha='left', va='center', fontsize=9)
    
    # Draw cliques (methods not significantly different)
    # Find groups where rank difference < CD
    cliques = []
    used = set()
    
    for i, (m1, r1) in enumerate(sorted_ranks):
        if m1 in used:
            continue
        clique = [m1]
        for j, (m2, r2) in enumerate(sorted_ranks[i+1:], i+1):
            if abs(r2 - r1) < cd:
                clique.append(m2)
        if len(clique) > 1:
            cliques.append((clique, r1, sorted_ranks[sorted_ranks.index((clique[-1], avg_ranks[clique[-1]]))][1]))
    
    # Draw clique bars
    clique_y = 0.78
    for clique, r_start, r_end in cliques:
        ax.plot([r_start, r_end], [clique_y, clique_y], 'k-', linewidth=3)
        clique_y -= 0.03
    
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.axis('off')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logging.info(f"CD diagram saved to {output_path}")
    
    plt.close()
    
    return cd


def generate_statistical_report(df, level, cv_type, feature_set, output_dir, metric='holdout_mcc'):
    """
    Generate comprehensive statistical analysis report.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    report_lines = []
    report_lines.append(f"# Statistical Analysis Report")
    report_lines.append(f"## {level.title()} Level - {cv_type.title()} CV - {feature_set.title()} Features")
    report_lines.append(f"\n**Metric:** {metric}")
    report_lines.append(f"**Number of projects (n):** {len(ALL_PROJECTS)}")
    report_lines.append("")
    
    # 1. ML Algorithm Comparison
    report_lines.append("---")
    report_lines.append("# 1. ML Algorithm Comparison")
    report_lines.append("")
    report_lines.append("For each project, we take the **best score across all resampling methods** for each ML algorithm.")
    report_lines.append("This provides a fair comparison of algorithms' best-case performance.")
    report_lines.append("")
    
    model_matrix = create_project_method_matrix(df, 'model', metric)
    
    # Friedman test
    friedman_result = friedman_test(model_matrix)
    
    if 'error' not in friedman_result:
        report_lines.append(f"### Friedman Test")
        report_lines.append(f"- **Chi-square statistic:** {friedman_result['statistic']:.4f}")
        report_lines.append(f"- **p-value:** {friedman_result['p_value']:.6f}")
        report_lines.append(f"- **Significant at α=0.05:** {'Yes' if friedman_result['p_value'] < 0.05 else 'No'}")
        report_lines.append("")
        
        # Average ranks
        report_lines.append("### Average Ranks (lower is better)")
        sorted_ranks = sorted(friedman_result['avg_ranks'].items(), key=lambda x: x[1])
        report_lines.append("| Rank | Algorithm | Avg. Rank |")
        report_lines.append("|------|-----------|-----------|")
        for i, (algo, rank) in enumerate(sorted_ranks, 1):
            report_lines.append(f"| {i} | {algo} | {rank:.3f} |")
        report_lines.append("")
        
        # CD Diagram
        cd = critical_difference_diagram(
            friedman_result['avg_ranks'],
            friedman_result['n_projects'],
            friedman_result['n_methods'],
            title=f"ML Algorithms - {level.title()} Level ({cv_type}, {feature_set})",
            output_path=output_dir / f"cd_diagram_models_{metric}.png"
        )
        report_lines.append(f"**Critical Difference (CD):** {cd:.3f}")
        report_lines.append(f"\n![CD Diagram - Models](cd_diagram_models_{metric}.png)")
        report_lines.append("")
        
        # Post-hoc tests
        if friedman_result['p_value'] < 0.05:
            report_lines.append("### Post-hoc Analysis (Nemenyi)")
            nemenyi_p = nemenyi_posthoc(model_matrix)
            if nemenyi_p is not None:
                # Find significant differences
                sig_pairs = []
                for i, m1 in enumerate(nemenyi_p.index):
                    for j, m2 in enumerate(nemenyi_p.columns):
                        if i < j and nemenyi_p.iloc[i, j] < 0.05:
                            sig_pairs.append((m1, m2, nemenyi_p.iloc[i, j]))
                
                if sig_pairs:
                    report_lines.append("\n**Significant differences (p < 0.05):**")
                    report_lines.append("| Algorithm 1 | Algorithm 2 | p-value |")
                    report_lines.append("|-------------|-------------|---------|")
                    for m1, m2, p in sorted(sig_pairs, key=lambda x: x[2]):
                        report_lines.append(f"| {m1} | {m2} | {p:.4f} |")
                else:
                    report_lines.append("\nNo significant pairwise differences found.")
            report_lines.append("")
            
            # Holm-Bonferroni Wilcoxon (more powerful)
            report_lines.append("### Post-hoc Analysis (Holm-Bonferroni corrected Wilcoxon)")
            wilcoxon_results = holm_bonferroni_wilcoxon(model_matrix)
            if not wilcoxon_results.empty:
                sig_wilcoxon = wilcoxon_results[wilcoxon_results['significant']]
                if not sig_wilcoxon.empty:
                    report_lines.append("\n**Significant differences (adjusted p < 0.05):**")
                    report_lines.append("| Algorithm 1 | Algorithm 2 | p-value | Adjusted p |")
                    report_lines.append("|-------------|-------------|---------|------------|")
                    for _, row in sig_wilcoxon.iterrows():
                        report_lines.append(f"| {row['method1']} | {row['method2']} | {row['p_value']:.4f} | {row['p_adjusted']:.4f} |")
                else:
                    report_lines.append("\nNo significant pairwise differences after Holm-Bonferroni correction.")
            report_lines.append("")
        
        # Effect sizes
        report_lines.append("### Effect Sizes (Cliff's Delta)")
        effect_sizes = calculate_effect_sizes(model_matrix)
        large_effects = effect_sizes[effect_sizes['effect_size'].isin(['large', 'medium'])]
        if not large_effects.empty:
            report_lines.append("\n**Medium to Large effects:**")
            report_lines.append("| Algorithm 1 | Algorithm 2 | Cliff's δ | Effect Size | Favors |")
            report_lines.append("|-------------|-------------|-----------|-------------|--------|")
            for _, row in large_effects.sort_values('cliffs_delta', key=abs, ascending=False).head(15).iterrows():
                report_lines.append(f"| {row['method1']} | {row['method2']} | {row['cliffs_delta']:.3f} | {row['effect_size']} | {row['favors']} |")
        report_lines.append("")
    
    # 2. Resampling Method Comparison
    report_lines.append("---")
    report_lines.append("# 2. Resampling Method Comparison")
    report_lines.append("")
    report_lines.append("For each project, we take the **best score across all ML algorithms** for each resampling method.")
    report_lines.append("")
    
    resampling_matrix = create_project_method_matrix(df, 'resampling', metric)
    friedman_resampling = friedman_test(resampling_matrix)
    
    if 'error' not in friedman_resampling:
        report_lines.append(f"### Friedman Test")
        report_lines.append(f"- **Chi-square statistic:** {friedman_resampling['statistic']:.4f}")
        report_lines.append(f"- **p-value:** {friedman_resampling['p_value']:.6f}")
        report_lines.append(f"- **Significant at α=0.05:** {'Yes' if friedman_resampling['p_value'] < 0.05 else 'No'}")
        report_lines.append("")
        
        # Average ranks
        report_lines.append("### Average Ranks (lower is better)")
        sorted_ranks = sorted(friedman_resampling['avg_ranks'].items(), key=lambda x: x[1])
        report_lines.append("| Rank | Resampling | Avg. Rank |")
        report_lines.append("|------|------------|-----------|")
        for i, (method, rank) in enumerate(sorted_ranks, 1):
            report_lines.append(f"| {i} | {method} | {rank:.3f} |")
        report_lines.append("")
        
        # CD Diagram
        cd = critical_difference_diagram(
            friedman_resampling['avg_ranks'],
            friedman_resampling['n_projects'],
            friedman_resampling['n_methods'],
            title=f"Resampling Methods - {level.title()} Level ({cv_type}, {feature_set})",
            output_path=output_dir / f"cd_diagram_resampling_{metric}.png"
        )
        report_lines.append(f"**Critical Difference (CD):** {cd:.3f}")
        report_lines.append(f"\n![CD Diagram - Resampling](cd_diagram_resampling_{metric}.png)")
        report_lines.append("")
        
        # Effect sizes for resampling
        report_lines.append("### Effect Sizes (Cliff's Delta)")
        effect_sizes_resampling = calculate_effect_sizes(resampling_matrix)
        large_effects = effect_sizes_resampling[effect_sizes_resampling['effect_size'].isin(['large', 'medium'])]
        if not large_effects.empty:
            report_lines.append("\n**Medium to Large effects:**")
            report_lines.append("| Method 1 | Method 2 | Cliff's δ | Effect Size | Favors |")
            report_lines.append("|----------|----------|-----------|-------------|--------|")
            for _, row in large_effects.sort_values('cliffs_delta', key=abs, ascending=False).head(10).iterrows():
                report_lines.append(f"| {row['method1']} | {row['method2']} | {row['cliffs_delta']:.3f} | {row['effect_size']} | {row['favors']} |")
        report_lines.append("")
    
    # 3. Summary Statistics
    report_lines.append("---")
    report_lines.append("# 3. Descriptive Statistics")
    report_lines.append("")
    
    # Best per project
    report_lines.append("### Best Configuration per Project")
    report_lines.append("| Project | Best Model | Best Resampling | Holdout MCC | Holdout F1 |")
    report_lines.append("|---------|------------|-----------------|-------------|------------|")
    
    best_per_project = df.loc[df.groupby('project')['holdout_mcc'].idxmax()]
    for _, row in best_per_project.sort_values('holdout_mcc', ascending=False).iterrows():
        report_lines.append(f"| {row['project']} | {row['model']} | {row['resampling']} | {row['holdout_mcc']:.4f} | {row['holdout_f1']:.4f} |")
    report_lines.append("")
    
    # Overall statistics
    report_lines.append("### Aggregate Statistics (Macro-averaged across projects)")
    report_lines.append("")
    
    # For models
    model_stats = model_matrix.agg(['mean', 'std', 'median']).T
    model_stats = model_stats.sort_values('mean', ascending=False)
    report_lines.append("**ML Algorithms (best per project):**")
    report_lines.append("| Algorithm | Mean | Std | Median |")
    report_lines.append("|-----------|------|-----|--------|")
    for algo in model_stats.index:
        report_lines.append(f"| {algo} | {model_stats.loc[algo, 'mean']:.4f} | {model_stats.loc[algo, 'std']:.4f} | {model_stats.loc[algo, 'median']:.4f} |")
    report_lines.append("")
    
    # For resampling
    resampling_stats = resampling_matrix.agg(['mean', 'std', 'median']).T
    resampling_stats = resampling_stats.sort_values('mean', ascending=False)
    report_lines.append("**Resampling Methods (best per project):**")
    report_lines.append("| Method | Mean | Std | Median |")
    report_lines.append("|--------|------|-----|--------|")
    for method in resampling_stats.index:
        report_lines.append(f"| {method} | {resampling_stats.loc[method, 'mean']:.4f} | {resampling_stats.loc[method, 'std']:.4f} | {resampling_stats.loc[method, 'median']:.4f} |")
    report_lines.append("")
    
    # Save report
    report_path = output_dir / "statistical_analysis.md"
    with open(report_path, 'w') as f:
        f.write('\n'.join(report_lines))
    
    logging.info(f"Statistical report saved to {report_path}")
    
    return '\n'.join(report_lines)


def generate_box_plots(df, level, cv_type, feature_set, output_dir, metric='holdout_mcc'):
    """Generate box plots for visual comparison."""
    output_dir = Path(output_dir)
    
    # 1. Box plot by ML Algorithm
    model_matrix = create_project_method_matrix(df, 'model', metric)
    
    plt.figure(figsize=(14, 6))
    model_matrix_melted = model_matrix.melt(var_name='Algorithm', value_name=metric)
    
    # Order by median
    order = model_matrix.median().sort_values(ascending=False).index
    
    ax = sns.boxplot(data=model_matrix_melted, x='Algorithm', y=metric, order=order, palette='viridis')
    sns.stripplot(data=model_matrix_melted, x='Algorithm', y=metric, order=order, 
                  color='black', alpha=0.5, size=4)
    
    plt.xticks(rotation=45, ha='right')
    plt.xlabel('ML Algorithm', fontsize=12)
    plt.ylabel(f'{metric.replace("_", " ").title()}', fontsize=12)
    plt.title(f'ML Algorithm Performance Distribution\n({level.title()} Level, {cv_type}, {feature_set})', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_dir / f'boxplot_models_{metric}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Box plot by Resampling
    resampling_matrix = create_project_method_matrix(df, 'resampling', metric)
    
    plt.figure(figsize=(14, 6))
    resampling_melted = resampling_matrix.melt(var_name='Resampling', value_name=metric)
    
    order = resampling_matrix.median().sort_values(ascending=False).index
    
    ax = sns.boxplot(data=resampling_melted, x='Resampling', y=metric, order=order, palette='coolwarm')
    sns.stripplot(data=resampling_melted, x='Resampling', y=metric, order=order,
                  color='black', alpha=0.5, size=4)
    
    plt.xticks(rotation=45, ha='right')
    plt.xlabel('Resampling Method', fontsize=12)
    plt.ylabel(f'{metric.replace("_", " ").title()}', fontsize=12)
    plt.title(f'Resampling Method Performance Distribution\n({level.title()} Level, {cv_type}, {feature_set})', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_dir / f'boxplot_resampling_{metric}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    logging.info(f"Box plots saved to {output_dir}")


def generate_heatmap(df, level, cv_type, feature_set, output_dir, metric='holdout_mcc'):
    """Generate heatmap of project × method performance."""
    output_dir = Path(output_dir)
    
    # Model heatmap
    model_matrix = create_project_method_matrix(df, 'model', metric)
    
    # Sort columns by mean performance
    col_order = model_matrix.mean().sort_values(ascending=False).index
    # Sort rows by mean performance
    row_order = model_matrix.mean(axis=1).sort_values(ascending=False).index
    
    model_matrix = model_matrix.loc[row_order, col_order]
    
    plt.figure(figsize=(14, 10))
    sns.heatmap(model_matrix, annot=True, fmt='.3f', cmap='RdYlGn', 
                center=model_matrix.mean().mean(),
                linewidths=0.5, annot_kws={'size': 8})
    plt.xlabel('ML Algorithm', fontsize=12)
    plt.ylabel('Project', fontsize=12)
    plt.title(f'Project × Algorithm Performance Heatmap ({metric})\n({level.title()} Level, {cv_type}, {feature_set})', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_dir / f'heatmap_models_{metric}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Resampling heatmap
    resampling_matrix = create_project_method_matrix(df, 'resampling', metric)
    
    col_order = resampling_matrix.mean().sort_values(ascending=False).index
    row_order = resampling_matrix.mean(axis=1).sort_values(ascending=False).index
    resampling_matrix = resampling_matrix.loc[row_order, col_order]
    
    plt.figure(figsize=(14, 10))
    sns.heatmap(resampling_matrix, annot=True, fmt='.3f', cmap='RdYlGn',
                center=resampling_matrix.mean().mean(),
                linewidths=0.5, annot_kws={'size': 8})
    plt.xlabel('Resampling Method', fontsize=12)
    plt.ylabel('Project', fontsize=12)
    plt.title(f'Project × Resampling Performance Heatmap ({metric})\n({level.title()} Level, {cv_type}, {feature_set})', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_dir / f'heatmap_resampling_{metric}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    logging.info(f"Heatmaps saved to {output_dir}")


def compare_cv_types(level, feature_set='full', metric='holdout_mcc'):
    """
    Compare temporal vs shuffle CV using paired Wilcoxon test.
    """
    df_temporal = collect_all_results(level, 'temporal', feature_set)
    df_shuffle = collect_all_results(level, 'shuffle', feature_set)
    
    # Get best per project for each CV type
    best_temporal = df_temporal.loc[df_temporal.groupby('project')[metric].idxmax()][['project', metric]]
    best_temporal = best_temporal.rename(columns={metric: 'temporal'})
    
    best_shuffle = df_shuffle.loc[df_shuffle.groupby('project')[metric].idxmax()][['project', metric]]
    best_shuffle = best_shuffle.rename(columns={metric: 'shuffle'})
    
    comparison = best_temporal.merge(best_shuffle, on='project')
    
    # Wilcoxon test
    stat, p_value = wilcoxon(comparison['temporal'], comparison['shuffle'])
    
    # Effect size
    delta, interp = cliffs_delta(comparison['temporal'].values, comparison['shuffle'].values)
    
    return {
        'comparison_df': comparison,
        'wilcoxon_stat': stat,
        'p_value': p_value,
        'cliffs_delta': delta,
        'effect_size': interp,
        'temporal_mean': comparison['temporal'].mean(),
        'shuffle_mean': comparison['shuffle'].mean(),
        'temporal_median': comparison['temporal'].median(),
        'shuffle_median': comparison['shuffle'].median()
    }


def compare_feature_sets(level, cv_type='temporal', metric='holdout_mcc'):
    """
    Compare full vs no_go_metrics feature sets using paired Wilcoxon test.
    """
    df_full = collect_all_results(level, cv_type, 'full')
    df_no_go = collect_all_results(level, cv_type, 'no_go_metrics')
    
    if df_no_go.empty:
        return {'error': 'no_go_metrics results not found'}
    
    # Get best per project for each feature set
    best_full = df_full.loc[df_full.groupby('project')[metric].idxmax()][['project', metric]]
    best_full = best_full.rename(columns={metric: 'full'})
    
    best_no_go = df_no_go.loc[df_no_go.groupby('project')[metric].idxmax()][['project', metric]]
    best_no_go = best_no_go.rename(columns={metric: 'no_go_metrics'})
    
    comparison = best_full.merge(best_no_go, on='project')
    
    if len(comparison) < 3:
        return {'error': 'Not enough data for comparison'}
    
    # Wilcoxon test
    stat, p_value = wilcoxon(comparison['full'], comparison['no_go_metrics'])
    
    # Effect size
    delta, interp = cliffs_delta(comparison['full'].values, comparison['no_go_metrics'].values)
    
    return {
        'comparison_df': comparison,
        'wilcoxon_stat': stat,
        'p_value': p_value,
        'cliffs_delta': delta,
        'effect_size': interp,
        'full_mean': comparison['full'].mean(),
        'no_go_mean': comparison['no_go_metrics'].mean()
    }


def main():
    parser = argparse.ArgumentParser(description='Statistical Analysis for Bug Prediction Study')
    parser.add_argument('--level', type=str, default='commit', choices=ALL_LEVELS,
                        help='Analysis level (commit, file, method)')
    parser.add_argument('--cv-type', type=str, default='temporal', choices=['temporal', 'shuffle'],
                        help='Cross-validation type')
    parser.add_argument('--feature-set', type=str, default='full', choices=['full', 'no_go_metrics'],
                        help='Feature set to analyze')
    parser.add_argument('--metric', type=str, default='holdout_mcc',
                        choices=['holdout_mcc', 'holdout_f1', 'holdout_pr_auc', 'holdout_roc_auc'],
                        help='Metric for comparison')
    parser.add_argument('--all-levels', action='store_true',
                        help='Run analysis for all levels')
    parser.add_argument('--compare-cv', action='store_true',
                        help='Compare temporal vs shuffle CV')
    parser.add_argument('--compare-features', action='store_true',
                        help='Compare full vs no_go_metrics features')
    parser.add_argument('--quality-filter', type=str, default='primary',
                        choices=['primary', 'exploratory', 'all'],
                        help='Dataset quality filter: primary (only statistically reliable projects), '
                             'exploratory (include smaller projects), all (no filter)')
    
    args = parser.parse_args()
    
    levels_to_analyze = ALL_LEVELS if args.all_levels else [args.level]
    
    for level in levels_to_analyze:
        logging.info(f"\n{'='*60}")
        logging.info(f"Analyzing level: {level}")
        logging.info(f"Quality filter: {args.quality_filter}")
        logging.info(f"{'='*60}")
        
        # Collect data with quality filter
        df = collect_all_results(level, args.cv_type, args.feature_set, 
                                 quality_filter=args.quality_filter,
                                 include_quality_info=True)
        
        if df.empty:
            logging.warning(f"No data found for level={level}, cv_type={args.cv_type}, feature_set={args.feature_set}")
            continue
        
        logging.info(f"Collected {len(df)} results from {df['project'].nunique()} projects")
        
        # Output directory - all outputs go to academic_outputs
        quality_suffix = f"_quality_{args.quality_filter}" if args.quality_filter != 'all' else ""
        output_dir = Path("academic_outputs") / "statistical_analysis" / level / args.cv_type / f"{args.feature_set}{quality_suffix}"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate statistical report
        generate_statistical_report(df, level, args.cv_type, args.feature_set, output_dir, args.metric)
        
        # Generate visualizations
        generate_box_plots(df, level, args.cv_type, args.feature_set, output_dir, args.metric)
        generate_heatmap(df, level, args.cv_type, args.feature_set, output_dir, args.metric)
        
        # Additional comparisons
        if args.compare_cv:
            logging.info("\nComparing CV types (temporal vs shuffle)...")
            cv_comparison = compare_cv_types(level, args.feature_set, args.metric)
            if 'error' not in cv_comparison:
                logging.info(f"  Wilcoxon p-value: {cv_comparison['p_value']:.6f}")
                logging.info(f"  Cliff's delta: {cv_comparison['cliffs_delta']:.3f} ({cv_comparison['effect_size']})")
                logging.info(f"  Temporal mean: {cv_comparison['temporal_mean']:.4f}")
                logging.info(f"  Shuffle mean: {cv_comparison['shuffle_mean']:.4f}")
        
        if args.compare_features:
            logging.info("\nComparing feature sets (full vs no_go_metrics)...")
            feature_comparison = compare_feature_sets(level, args.cv_type, args.metric)
            if 'error' not in feature_comparison:
                logging.info(f"  Wilcoxon p-value: {feature_comparison['p_value']:.6f}")
                logging.info(f"  Cliff's delta: {feature_comparison['cliffs_delta']:.3f} ({feature_comparison['effect_size']})")
                logging.info(f"  Full features mean: {feature_comparison['full_mean']:.4f}")
                logging.info(f"  No Go metrics mean: {feature_comparison['no_go_mean']:.4f}")
    
    logging.info("\n" + "="*60)
    logging.info("Statistical analysis complete!")
    logging.info("="*60)


if __name__ == '__main__':
    main()
