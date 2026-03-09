#!/usr/bin/env python3
"""
Academic Figures and Tables Generator
======================================

Generates publication-ready figures and tables for the bug prediction study:
1. Dataset Statistics Table
2. Level Comparison (commit vs file vs method) - by metric and CV type
3. Temporal vs Shuffle CV Comparison
4. Feature Importance Analysis (Go metrics impact)
5. Cross-level Performance Summary
6. Best Configuration Summary Table
7. Nemenyi Post-hoc P-value Tables
8. Model/Resampling Performance Heatmaps
9. Comprehensive Box Plots by Metric
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
from scipy.stats import wilcoxon, friedmanchisquare, rankdata
import scikit_posthocs as sp
import argparse
import logging

warnings.filterwarnings('ignore')

# All metrics to analyze
ALL_METRICS = ['holdout_mcc', 'holdout_f1', 'holdout_pr_auc', 'holdout_roc_auc', 
               'holdout_precision', 'holdout_recall']
CV_METRICS = ['cv_mcc', 'cv_f1', 'cv_pr_auc', 'cv_roc_auc']

METRIC_LABELS = {
    'holdout_mcc': 'MCC (Holdout)',
    'holdout_f1': 'F1 (Holdout)',
    'holdout_pr_auc': 'PR-AUC (Holdout)',
    'holdout_roc_auc': 'ROC-AUC (Holdout)',
    'holdout_precision': 'Precision (Holdout)',
    'holdout_recall': 'Recall (Holdout)',
    'cv_mcc': 'MCC (CV)',
    'cv_f1': 'F1 (CV)',
    'cv_pr_auc': 'PR-AUC (CV)',
    'cv_roc_auc': 'ROC-AUC (CV)',
}

# Short labels for figures
METRIC_SHORT_LABELS = {
    'holdout_mcc': 'MCC',
    'holdout_f1': 'F1',
    'holdout_pr_auc': 'PR-AUC',
    'holdout_roc_auc': 'ROC-AUC',
    'cv_mcc': 'MCC',
    'cv_f1': 'F1',
    'cv_pr_auc': 'PR-AUC',
    'cv_roc_auc': 'ROC-AUC',
}

# Score types for analysis
SCORE_TYPES = ['holdout', 'cv']  # holdout test set vs cross-validation
PRIMARY_METRICS = ['mcc', 'f1']  # MCC and F1 as primary metrics

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
ALL_RESAMPLING = ["none", "smote", "random_under", "near_miss", "tomek", 
                  "random_over", "adasyn", "borderline", "smote_tomek", "smote_enn", "rose"]
ALL_MODELS = [
    'naive_bayes', 'xgboost', 'random_forest', 'logistic_regression',
    'catboost', 'lightgbm', 'gradient_boosting', 'decision_tree',
    'voting', 'mlp', 'stacking'
]

# Plotting style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['figure.titlesize'] = 14


def get_results_dir(level):
    """Get the results directory for a specific level."""
    return BASE_DIR / f"results_{level}_level"


def get_data_dir(level):
    """Get the data directory for a specific level."""
    if level == 'commit':
        return BASE_DIR / 'commit_data'
    elif level == 'file':
        return BASE_DIR / 'file_data'
    else:
        return BASE_DIR / 'method_data'


def collect_all_results(level, cv_type='temporal', feature_set='full'):
    """Collect all results from analysis_summary.json files."""
    results_dir = get_results_dir(level)
    all_data = []
    
    for project in ALL_PROJECTS:
        feature_set_dir = results_dir / project / cv_type / feature_set
        
        if not feature_set_dir.exists():
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
                
                for model_name, model_data in models.items():
                    cv_metrics = model_data.get('cv_metrics', {})
                    holdout_metrics = model_data.get('holdout_metrics', {})
                    
                    all_data.append({
                        'project': project,
                        'resampling': resampling,
                        'model': model_name,
                        'cv_mcc': cv_metrics.get('mcc'),
                        'cv_mcc_std': cv_metrics.get('mcc_std'),
                        'cv_f1': cv_metrics.get('f1_bug'),
                        'cv_f1_std': cv_metrics.get('f1_bug_std'),
                        'cv_pr_auc': cv_metrics.get('pr_auc'),
                        'cv_roc_auc': cv_metrics.get('roc_auc'),
                        'holdout_mcc': holdout_metrics.get('mcc'),
                        'holdout_f1': holdout_metrics.get('f1_bug'),
                        'holdout_pr_auc': holdout_metrics.get('pr_auc'),
                        'holdout_roc_auc': holdout_metrics.get('roc_auc'),
                        'holdout_precision': holdout_metrics.get('precision_bug'),
                        'holdout_recall': holdout_metrics.get('recall_bug'),
                    })
            except Exception as e:
                logging.error(f"Error reading {summary_file}: {e}")
    
    return pd.DataFrame(all_data)


def generate_dataset_statistics_table(output_dir):
    """
    Generate Table 1: Dataset Statistics
    Shows sample counts, bug ratios, and features for each project and level.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    stats_data = []
    
    for level in ALL_LEVELS:
        results_dir = get_results_dir(level)
        
        for project in ALL_PROJECTS:
            # Try to find any analysis_summary.json for this project
            project_dir = results_dir / project / "temporal" / "full"
            
            if not project_dir.exists():
                continue
            
            # Find first resampling dir with data
            for resampling_dir in project_dir.iterdir():
                if not resampling_dir.is_dir():
                    continue
                
                summary_file = resampling_dir / "analysis_summary.json"
                if summary_file.exists():
                    try:
                        with open(summary_file, 'r') as f:
                            data = json.load(f)
                        
                        dataset_info = data.get('dataset_info', {})
                        
                        stats_data.append({
                            'Project': project,
                            'Level': level.title(),
                            'Total Samples': dataset_info.get('n_total_samples', 'N/A'),
                            'Train+Val': dataset_info.get('n_train_val_samples', 'N/A'),
                            'Holdout': dataset_info.get('n_holdout_samples', 'N/A'),
                            'Bug Ratio (Train)': f"{dataset_info.get('train_val_bug_ratio', 0)*100:.1f}%",
                            'Bug Ratio (Holdout)': f"{dataset_info.get('holdout_bug_ratio', 0)*100:.1f}%",
                            'Features': dataset_info.get('n_features', 'N/A'),
                        })
                        break
                    except Exception as e:
                        logging.error(f"Error: {e}")
    
    df = pd.DataFrame(stats_data)
    
    # Create pivot table for each level
    markdown_lines = ["# Table 1: Dataset Statistics\n"]
    
    for level in ALL_LEVELS:
        level_df = df[df['Level'] == level.title()].drop('Level', axis=1)
        
        if level_df.empty:
            continue
        
        markdown_lines.append(f"\n## {level.title()} Level\n")
        markdown_lines.append(level_df.to_markdown(index=False))
        markdown_lines.append("\n")
    
    # Save markdown
    with open(output_dir / "table1_dataset_statistics.md", 'w') as f:
        f.write('\n'.join(markdown_lines))
    
    # Also save as CSV
    df.to_csv(output_dir / "dataset_statistics.csv", index=False)
    
    logging.info(f"Dataset statistics saved to {output_dir}")
    
    # Create summary figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for i, level in enumerate(ALL_LEVELS):
        level_df = df[df['Level'] == level.title()].copy()
        if level_df.empty:
            continue
        
        level_df['Total Samples'] = pd.to_numeric(level_df['Total Samples'], errors='coerce')
        level_df = level_df.dropna(subset=['Total Samples'])
        level_df = level_df.sort_values('Total Samples', ascending=True)
        
        ax = axes[i]
        colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(level_df)))
        bars = ax.barh(level_df['Project'], level_df['Total Samples'], color=colors)
        ax.set_xlabel('Number of Samples')
        ax.set_title(f'{level.title()} Level')
        ax.set_xlim(0, level_df['Total Samples'].max() * 1.1)
        
        # Add value labels
        for bar, val in zip(bars, level_df['Total Samples']):
            ax.text(val + level_df['Total Samples'].max()*0.02, bar.get_y() + bar.get_height()/2, 
                   f'{int(val):,}', va='center', fontsize=8)
    
    plt.suptitle('Dataset Size by Project and Level', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / "figure_dataset_sizes.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    return df


def generate_level_comparison_table(output_dir, cv_type='temporal', feature_set='full'):
    """
    Generate Table 2: Performance Comparison Across Levels
    Compares commit, file, and method level results.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    comparison_data = []
    
    for level in ALL_LEVELS:
        df = collect_all_results(level, cv_type, feature_set)
        
        if df.empty:
            continue
        
        # Get best per project
        best_per_project = df.loc[df.groupby('project')['holdout_mcc'].idxmax()]
        
        comparison_data.append({
            'Level': level.title(),
            'Mean MCC': best_per_project['holdout_mcc'].mean(),
            'Std MCC': best_per_project['holdout_mcc'].std(),
            'Median MCC': best_per_project['holdout_mcc'].median(),
            'Mean F1': best_per_project['holdout_f1'].mean(),
            'Std F1': best_per_project['holdout_f1'].std(),
            'Mean PR-AUC': best_per_project['holdout_pr_auc'].mean(),
            'Mean ROC-AUC': best_per_project['holdout_roc_auc'].mean(),
            'Best Model': best_per_project['model'].mode().iloc[0] if not best_per_project['model'].mode().empty else 'N/A',
            'Best Resampling': best_per_project['resampling'].mode().iloc[0] if not best_per_project['resampling'].mode().empty else 'N/A',
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    
    # Generate markdown table
    markdown_lines = ["# Table 2: Performance Comparison Across Levels\n"]
    markdown_lines.append(f"CV Type: {cv_type.title()}, Feature Set: {feature_set}\n")
    markdown_lines.append("\n| Level | MCC (Mean±Std) | Median MCC | F1 (Mean±Std) | PR-AUC | ROC-AUC | Most Frequent Model | Most Frequent Resampling |")
    markdown_lines.append("|-------|----------------|------------|---------------|--------|---------|---------------------|--------------------------|")
    
    for _, row in comparison_df.iterrows():
        markdown_lines.append(
            f"| {row['Level']} | {row['Mean MCC']:.4f}±{row['Std MCC']:.4f} | {row['Median MCC']:.4f} | "
            f"{row['Mean F1']:.4f}±{row['Std F1']:.4f} | {row['Mean PR-AUC']:.4f} | {row['Mean ROC-AUC']:.4f} | "
            f"{row['Best Model']} | {row['Best Resampling']} |"
        )
    
    with open(output_dir / "table2_level_comparison.md", 'w') as f:
        f.write('\n'.join(markdown_lines))
    
    comparison_df.to_csv(output_dir / "level_comparison.csv", index=False)
    
    # Statistical test: Friedman test across levels
    # Collect best MCC per project for each level
    level_scores = {}
    for level in ALL_LEVELS:
        df = collect_all_results(level, cv_type, feature_set)
        if not df.empty:
            best = df.loc[df.groupby('project')['holdout_mcc'].idxmax()][['project', 'holdout_mcc']]
            level_scores[level] = best.set_index('project')['holdout_mcc']
    
    if len(level_scores) == 3:
        # Align by project
        combined = pd.DataFrame(level_scores)
        combined = combined.dropna()
        
        if len(combined) >= 3:
            stat, p_value = friedmanchisquare(
                combined['commit'].values,
                combined['file'].values,
                combined['method'].values
            )
            k = len(combined.columns)
            df_friedman = k - 1
            
            markdown_lines.append(f"\n\n## Statistical Comparison (Friedman Test)")
            markdown_lines.append(f"- Friedman: χ²({df_friedman}) = {stat:.4f}, p = {p_value:.6f}")
            markdown_lines.append(f"- Significant at α=0.05: {'Yes' if p_value < 0.05 else 'No'}")
            
            # Pairwise Wilcoxon tests
            markdown_lines.append("\n### Pairwise Wilcoxon Tests:")
            for l1, l2 in [('commit', 'file'), ('commit', 'method'), ('file', 'method')]:
                stat_w, p_w = wilcoxon(combined[l1], combined[l2])
                markdown_lines.append(f"- {l1} vs {l2}: p={p_w:.4f}")
            
            with open(output_dir / "table2_level_comparison.md", 'w') as f:
                f.write('\n'.join(markdown_lines))
    
    # Create comparison figure
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Box plot of MCC by level
    level_data = []
    for level in ALL_LEVELS:
        df = collect_all_results(level, cv_type, feature_set)
        if not df.empty:
            best = df.loc[df.groupby('project')['holdout_mcc'].idxmax()]
            for _, row in best.iterrows():
                level_data.append({'Level': level.title(), 'MCC': row['holdout_mcc']})
    
    level_df = pd.DataFrame(level_data)
    
    ax = axes[0]
    sns.boxplot(data=level_df, x='Level', y='MCC', ax=ax, palette='Set2')
    sns.stripplot(data=level_df, x='Level', y='MCC', ax=ax, color='black', alpha=0.5, size=6)
    ax.set_ylabel('Holdout MCC')
    ax.set_title('Performance Distribution by Level')
    
    # Bar plot of mean metrics
    ax = axes[1]
    x = np.arange(len(comparison_df))
    width = 0.2
    
    ax.bar(x - width*1.5, comparison_df['Mean MCC'], width, label='MCC', color='#2ecc71')
    ax.bar(x - width*0.5, comparison_df['Mean F1'], width, label='F1', color='#3498db')
    ax.bar(x + width*0.5, comparison_df['Mean PR-AUC'], width, label='PR-AUC', color='#e74c3c')
    ax.bar(x + width*1.5, comparison_df['Mean ROC-AUC'], width, label='ROC-AUC', color='#9b59b6')
    
    ax.set_xlabel('Level')
    ax.set_ylabel('Score')
    ax.set_title('Mean Performance Metrics by Level')
    ax.set_xticks(x)
    ax.set_xticklabels(comparison_df['Level'])
    ax.legend(loc='lower right')
    ax.set_ylim(0, 1)
    
    plt.suptitle(f'Level Comparison ({cv_type.title()} CV)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / "figure_level_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    logging.info(f"Level comparison saved to {output_dir}")
    
    return comparison_df


def generate_cv_comparison(output_dir, feature_set='full'):
    """
    Generate Table 3: Temporal vs Shuffle CV Comparison
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    comparison_results = []
    
    markdown_lines = ["# Table 3: Temporal vs Shuffle CV Comparison\n"]
    markdown_lines.append("This table compares temporal (time-aware) and shuffle (random) cross-validation strategies.\n")
    
    for level in ALL_LEVELS:
        df_temporal = collect_all_results(level, 'temporal', feature_set)
        df_shuffle = collect_all_results(level, 'shuffle', feature_set)
        
        if df_temporal.empty or df_shuffle.empty:
            continue
        
        # Best per project for each CV type
        best_temporal = df_temporal.loc[df_temporal.groupby('project')['holdout_mcc'].idxmax()]
        best_shuffle = df_shuffle.loc[df_shuffle.groupby('project')['holdout_mcc'].idxmax()]
        
        # Merge and compare
        temporal_scores = best_temporal[['project', 'holdout_mcc']].rename(columns={'holdout_mcc': 'temporal'})
        shuffle_scores = best_shuffle[['project', 'holdout_mcc']].rename(columns={'holdout_mcc': 'shuffle'})
        
        merged = temporal_scores.merge(shuffle_scores, on='project')
        
        if len(merged) < 3:
            continue
        
        # Wilcoxon test
        stat, p_value = wilcoxon(merged['temporal'], merged['shuffle'])
        
        # Effect size (Cliff's delta)
        n1, n2 = len(merged['temporal']), len(merged['shuffle'])
        more = sum(1 for t, s in zip(merged['temporal'], merged['shuffle']) if t > s)
        less = sum(1 for t, s in zip(merged['temporal'], merged['shuffle']) if t < s)
        delta = (more - less) / (n1 * n2) if n1 * n2 > 0 else 0
        
        comparison_results.append({
            'Level': level.title(),
            'Temporal Mean': merged['temporal'].mean(),
            'Temporal Std': merged['temporal'].std(),
            'Shuffle Mean': merged['shuffle'].mean(),
            'Shuffle Std': merged['shuffle'].std(),
            'Wilcoxon p': p_value,
            'Cliffs Delta': delta,
            'Significant': p_value < 0.05
        })
    
    comparison_df = pd.DataFrame(comparison_results)
    
    markdown_lines.append("\n| Level | Temporal (Mean±Std) | Shuffle (Mean±Std) | Wilcoxon p | Cliff's δ | Significant |")
    markdown_lines.append("|-------|---------------------|---------------------|------------|-----------|-------------|")
    
    for _, row in comparison_df.iterrows():
        sig = "✓" if row['Significant'] else "✗"
        markdown_lines.append(
            f"| {row['Level']} | {row['Temporal Mean']:.4f}±{row['Temporal Std']:.4f} | "
            f"{row['Shuffle Mean']:.4f}±{row['Shuffle Std']:.4f} | {row['Wilcoxon p']:.4f} | "
            f"{row['Cliffs Delta']:.3f} | {sig} |"
        )
    
    markdown_lines.append("\n\n**Interpretation:**")
    markdown_lines.append("- Positive Cliff's δ favors Temporal CV")
    markdown_lines.append("- Negative Cliff's δ favors Shuffle CV")
    markdown_lines.append("- |δ| < 0.147: negligible, < 0.33: small, < 0.474: medium, ≥ 0.474: large")
    
    with open(output_dir / "table3_cv_comparison.md", 'w') as f:
        f.write('\n'.join(markdown_lines))
    
    comparison_df.to_csv(output_dir / "cv_comparison.csv", index=False)
    
    # Create comparison figure
    fig, axes = plt.subplots(1, len(ALL_LEVELS), figsize=(15, 5))
    
    for i, level in enumerate(ALL_LEVELS):
        df_temporal = collect_all_results(level, 'temporal', feature_set)
        df_shuffle = collect_all_results(level, 'shuffle', feature_set)
        
        if df_temporal.empty or df_shuffle.empty:
            continue
        
        best_temporal = df_temporal.loc[df_temporal.groupby('project')['holdout_mcc'].idxmax()]
        best_shuffle = df_shuffle.loc[df_shuffle.groupby('project')['holdout_mcc'].idxmax()]
        
        temporal_scores = best_temporal.set_index('project')['holdout_mcc']
        shuffle_scores = best_shuffle.set_index('project')['holdout_mcc']
        
        # Scatter plot: temporal vs shuffle
        ax = axes[i]
        common_projects = set(temporal_scores.index) & set(shuffle_scores.index)
        
        t_vals = [temporal_scores[p] for p in common_projects]
        s_vals = [shuffle_scores[p] for p in common_projects]
        
        ax.scatter(t_vals, s_vals, alpha=0.7, s=80, c='steelblue')
        
        # Add diagonal line
        lims = [min(ax.get_xlim()[0], ax.get_ylim()[0]), max(ax.get_xlim()[1], ax.get_ylim()[1])]
        ax.plot(lims, lims, 'k--', alpha=0.5, label='Equal performance')
        
        # Add project labels
        for p in common_projects:
            ax.annotate(p, (temporal_scores[p], shuffle_scores[p]), fontsize=7, alpha=0.7)
        
        ax.set_xlabel('Temporal CV (MCC)')
        ax.set_ylabel('Shuffle CV (MCC)')
        ax.set_title(f'{level.title()} Level')
        ax.set_aspect('equal')
    
    plt.suptitle('Temporal vs Shuffle CV Performance', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / "figure_cv_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    logging.info(f"CV comparison saved to {output_dir}")
    
    return comparison_df


def generate_best_configuration_table(output_dir, cv_type='temporal', feature_set='full'):
    """
    Generate Table 4: Best Configuration per Project
    Comprehensive summary of best-performing configurations.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_best = []
    
    for level in ALL_LEVELS:
        df = collect_all_results(level, cv_type, feature_set)
        
        if df.empty:
            continue
        
        # Get best per project
        best_per_project = df.loc[df.groupby('project')['holdout_mcc'].idxmax()]
        
        for _, row in best_per_project.iterrows():
            all_best.append({
                'Project': row['project'],
                'Level': level.title(),
                'Model': row['model'],
                'Resampling': row['resampling'],
                'MCC': row['holdout_mcc'],
                'F1': row['holdout_f1'],
                'PR-AUC': row['holdout_pr_auc'],
                'Precision': row['holdout_precision'],
                'Recall': row['holdout_recall'],
            })
    
    best_df = pd.DataFrame(all_best)
    
    # Pivot for cross-level view
    pivot_mcc = best_df.pivot(index='Project', columns='Level', values='MCC')
    
    markdown_lines = ["# Table 4: Best Configuration per Project\n"]
    markdown_lines.append(f"CV Type: {cv_type.title()}, Feature Set: {feature_set}\n")
    
    # For each level
    for level in ALL_LEVELS:
        level_df = best_df[best_df['Level'] == level.title()].sort_values('MCC', ascending=False)
        
        if level_df.empty:
            continue
        
        markdown_lines.append(f"\n## {level.title()} Level\n")
        markdown_lines.append("| Project | Model | Resampling | MCC | F1 | PR-AUC | Precision | Recall |")
        markdown_lines.append("|---------|-------|------------|-----|-----|--------|-----------|--------|")
        
        for _, row in level_df.iterrows():
            markdown_lines.append(
                f"| {row['Project']} | {row['Model']} | {row['Resampling']} | "
                f"{row['MCC']:.4f} | {row['F1']:.4f} | {row['PR-AUC']:.4f} | "
                f"{row['Precision']:.4f} | {row['Recall']:.4f} |"
            )
    
    # Cross-level MCC comparison
    markdown_lines.append("\n## Cross-Level MCC Comparison\n")
    markdown_lines.append(pivot_mcc.to_markdown())
    
    with open(output_dir / "table4_best_configurations.md", 'w') as f:
        f.write('\n'.join(markdown_lines))
    
    best_df.to_csv(output_dir / "best_configurations.csv", index=False)
    
    # Create heatmap figure
    if not pivot_mcc.empty:
        plt.figure(figsize=(8, 10))
        sns.heatmap(pivot_mcc.sort_values('Commit' if 'Commit' in pivot_mcc.columns else pivot_mcc.columns[0], ascending=False),
                    annot=True, fmt='.3f', cmap='RdYlGn', center=0.4,
                    linewidths=0.5, cbar_kws={'label': 'Holdout MCC'})
        plt.xlabel('Level')
        plt.ylabel('Project')
        plt.title(f'Best MCC by Project and Level\n({cv_type.title()} CV, {feature_set})')
        plt.tight_layout()
        plt.savefig(output_dir / "figure_best_mcc_heatmap.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    logging.info(f"Best configurations saved to {output_dir}")
    
    return best_df


def generate_model_resampling_frequency_table(output_dir, cv_type='temporal', feature_set='full'):
    """
    Generate Table 5: Frequency of Best Model/Resampling Combinations
    Shows which methods win most often across projects.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    model_wins = defaultdict(lambda: defaultdict(int))
    resampling_wins = defaultdict(lambda: defaultdict(int))
    
    for level in ALL_LEVELS:
        df = collect_all_results(level, cv_type, feature_set)
        
        if df.empty:
            continue
        
        best_per_project = df.loc[df.groupby('project')['holdout_mcc'].idxmax()]
        
        for _, row in best_per_project.iterrows():
            model_wins[level][row['model']] += 1
            resampling_wins[level][row['resampling']] += 1
    
    # Create summary tables
    markdown_lines = ["# Table 5: Winning Frequency Analysis\n"]
    markdown_lines.append("How often each method achieves the best performance across projects.\n")
    
    # Model frequency
    markdown_lines.append("\n## Model Win Frequency\n")
    model_df = pd.DataFrame(model_wins).fillna(0).astype(int)
    model_df['Total'] = model_df.sum(axis=1)
    model_df = model_df.sort_values('Total', ascending=False)
    markdown_lines.append(model_df.to_markdown())
    
    # Resampling frequency
    markdown_lines.append("\n\n## Resampling Win Frequency\n")
    resampling_df = pd.DataFrame(resampling_wins).fillna(0).astype(int)
    resampling_df['Total'] = resampling_df.sum(axis=1)
    resampling_df = resampling_df.sort_values('Total', ascending=False)
    markdown_lines.append(resampling_df.to_markdown())
    
    with open(output_dir / "table5_winning_frequency.md", 'w') as f:
        f.write('\n'.join(markdown_lines))
    
    # Create bar chart
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Model wins
    ax = axes[0]
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(model_df)))
    model_df['Total'].plot(kind='barh', ax=ax, color=colors)
    ax.set_xlabel('Number of Wins')
    ax.set_title('Model Win Frequency (All Levels)')
    ax.invert_yaxis()
    
    # Resampling wins
    ax = axes[1]
    colors = plt.cm.plasma(np.linspace(0.2, 0.8, len(resampling_df)))
    resampling_df['Total'].plot(kind='barh', ax=ax, color=colors)
    ax.set_xlabel('Number of Wins')
    ax.set_title('Resampling Win Frequency (All Levels)')
    ax.invert_yaxis()
    
    plt.suptitle('Method Win Frequency Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / "figure_winning_frequency.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    logging.info(f"Winning frequency analysis saved to {output_dir}")
    
    return model_df, resampling_df


def generate_radar_chart(output_dir, cv_type='temporal', feature_set='full'):
    """
    Generate Figure: Radar chart comparing metrics across levels.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    metrics = ['MCC', 'F1', 'PR-AUC', 'ROC-AUC', 'Precision', 'Recall']
    level_metrics = {}
    
    for level in ALL_LEVELS:
        df = collect_all_results(level, cv_type, feature_set)
        
        if df.empty:
            continue
        
        best_per_project = df.loc[df.groupby('project')['holdout_mcc'].idxmax()]
        
        level_metrics[level] = {
            'MCC': best_per_project['holdout_mcc'].mean(),
            'F1': best_per_project['holdout_f1'].mean(),
            'PR-AUC': best_per_project['holdout_pr_auc'].mean(),
            'ROC-AUC': best_per_project['holdout_roc_auc'].mean(),
            'Precision': best_per_project['holdout_precision'].mean(),
            'Recall': best_per_project['holdout_recall'].mean(),
        }
    
    if not level_metrics:
        return
    
    # Create radar chart
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]  # Complete the loop
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
    
    colors = ['#e74c3c', '#3498db', '#2ecc71']
    
    for (level, values), color in zip(level_metrics.items(), colors):
        vals = [values[m] for m in metrics]
        vals += vals[:1]
        
        ax.plot(angles, vals, 'o-', linewidth=2, label=level.title(), color=color)
        ax.fill(angles, vals, alpha=0.15, color=color)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 1)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    ax.set_title(f'Performance Metrics by Level\n({cv_type.title()} CV)', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / "figure_radar_chart.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    logging.info(f"Radar chart saved to {output_dir}")


# =============================================================================
# NEW: Level Comparison by Metric (Temporal and Shuffle separately)
# =============================================================================

def generate_level_comparison_by_metric(output_dir):
    """
    Generate level comparison figures for each metric and CV type.
    Creates separate box plots for temporal and shuffle CV.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    metrics_to_plot = ['holdout_mcc', 'holdout_f1', 'holdout_pr_auc', 'holdout_roc_auc']
    cv_types = ['temporal', 'shuffle']
    
    for cv_type in cv_types:
        # Collect data for all levels
        all_data = []
        
        for level in ALL_LEVELS:
            df = collect_all_results(level, cv_type, 'full')
            if df.empty:
                continue
            
            # Best per project
            best_per_project = df.loc[df.groupby('project')['holdout_mcc'].idxmax()]
            
            for _, row in best_per_project.iterrows():
                for metric in metrics_to_plot:
                    all_data.append({
                        'Level': level.title(),
                        'Metric': METRIC_LABELS.get(metric, metric),
                        'Value': row[metric],
                        'Project': row['project']
                    })
        
        if not all_data:
            continue
        
        plot_df = pd.DataFrame(all_data)
        
        # Create 2x2 subplot for each metric
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        axes = axes.flatten()
        
        colors = {'Commit': '#e74c3c', 'File': '#3498db', 'Method': '#2ecc71'}
        
        for i, metric in enumerate(metrics_to_plot):
            ax = axes[i]
            metric_label = METRIC_LABELS.get(metric, metric)
            metric_data = plot_df[plot_df['Metric'] == metric_label]
            
            sns.boxplot(data=metric_data, x='Level', y='Value', ax=ax, 
                       palette=colors, order=['Commit', 'File', 'Method'])
            sns.stripplot(data=metric_data, x='Level', y='Value', ax=ax,
                         color='black', alpha=0.5, size=6, order=['Commit', 'File', 'Method'])
            
            ax.set_ylabel(metric_label)
            ax.set_xlabel('')
            ax.set_title(f'{metric_label}')
            
            # Add Friedman test result
            level_data = {}
            for level in ALL_LEVELS:
                level_df = metric_data[metric_data['Level'] == level.title()]
                if not level_df.empty:
                    level_data[level] = level_df.set_index('Project')['Value']
            
            if len(level_data) == 3:
                combined = pd.DataFrame(level_data).dropna()
                if len(combined) >= 3:
                    try:
                        stat, p_val = friedmanchisquare(
                            combined['commit'].values,
                            combined['file'].values,
                            combined['method'].values
                        )
                        k = len(combined.columns)
                        df_friedman = k - 1
                        sig_text = f'Friedman: χ²({df_friedman})={stat:.4f}, p={p_val:.4f}'
                        if p_val < 0.05:
                            sig_text += ' *'
                        if p_val < 0.01:
                            sig_text += '*'
                        ax.text(0.02, 0.98, sig_text, transform=ax.transAxes,
                               fontsize=9, va='top', ha='left',
                               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
                    except:
                        pass
        
        plt.suptitle(f'Level Comparison by Metric ({cv_type.title()} CV)', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_dir / f"figure_level_comparison_{cv_type}_all_metrics.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        logging.info(f"Level comparison by metric ({cv_type}) saved to {output_dir}")
    
    # Also create a combined figure showing both CV types side by side for MCC
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    for i, cv_type in enumerate(cv_types):
        ax = axes[i]
        all_data = []
        
        for level in ALL_LEVELS:
            df = collect_all_results(level, cv_type, 'full')
            if df.empty:
                continue
            
            best_per_project = df.loc[df.groupby('project')['holdout_mcc'].idxmax()]
            for _, row in best_per_project.iterrows():
                all_data.append({
                    'Level': level.title(),
                    'MCC': row['holdout_mcc'],
                    'Project': row['project']
                })
        
        if all_data:
            plot_df = pd.DataFrame(all_data)
            colors = {'Commit': '#e74c3c', 'File': '#3498db', 'Method': '#2ecc71'}
            
            sns.boxplot(data=plot_df, x='Level', y='MCC', ax=ax,
                       palette=colors, order=['Commit', 'File', 'Method'])
            sns.stripplot(data=plot_df, x='Level', y='MCC', ax=ax,
                         color='black', alpha=0.5, size=6, order=['Commit', 'File', 'Method'])
            
            ax.set_ylabel('Holdout MCC')
            ax.set_xlabel('Granularity Level')
            ax.set_title(f'{cv_type.title()} CV')
    
    plt.suptitle('Level Comparison: MCC by CV Type', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / "figure_level_comparison_mcc_both_cv.png", dpi=300, bbox_inches='tight')
    plt.close()


# =============================================================================
# NEW: Nemenyi Post-hoc P-value Tables (Multi-metric support)
# =============================================================================

def generate_nemenyi_tables(output_dir, cv_type='temporal', feature_set='full'):
    """
    Generate Nemenyi post-hoc test p-value tables for:
    1. Model comparison by level
    2. Resampling comparison by level
    
    Now supports both holdout and CV scores, and both MCC and F1 metrics.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Run for both score types (holdout and cv) and both metrics (mcc and f1)
    for score_type in SCORE_TYPES:
        for metric_name in PRIMARY_METRICS:
            metric_col = f"{score_type}_{metric_name}"
            metric_label = f"{metric_name.upper()} ({score_type.title()})"
            
            markdown_lines = [f"# Nemenyi Post-hoc Test Results ({metric_label})\n"]
            markdown_lines.append(f"CV Type: {cv_type.title()}, Feature Set: {feature_set}\n")
            markdown_lines.append(f"Metric: {metric_label}\n")
            markdown_lines.append("P-values < 0.05 indicate statistically significant differences.\n")
            
            for level in ALL_LEVELS:
                df = collect_all_results(level, cv_type, feature_set)
                if df.empty or metric_col not in df.columns:
                    continue
                
                markdown_lines.append(f"\n## {level.title()} Level\n")
                
                # Model comparison
                markdown_lines.append("\n### Model Comparison\n")
                
                # Create project x model matrix (best across resampling)
                model_matrix = df.groupby(['project', 'model'])[metric_col].max().unstack()
                model_matrix = model_matrix.dropna(axis=1, how='all').dropna()
                
                if len(model_matrix) >= 3 and len(model_matrix.columns) >= 2:
                    try:
                        # Friedman test first
                        values = [model_matrix[col].values for col in model_matrix.columns]
                        stat, p_friedman = friedmanchisquare(*values)
                        k = len(model_matrix.columns)  # number of groups
                        df_friedman = k - 1  # degrees of freedom
                        markdown_lines.append(f"**Friedman test:** χ²({df_friedman}) = {stat:.4f}, p = {p_friedman:.6f}\n")
                        
                        if p_friedman < 0.05:
                            # Nemenyi post-hoc
                            p_values = sp.posthoc_nemenyi_friedman(model_matrix.values)
                            p_values.index = model_matrix.columns
                            p_values.columns = model_matrix.columns
                            
                            # Format as markdown table
                            markdown_lines.append("\n**Nemenyi Post-hoc P-values:**\n")
                            markdown_lines.append(p_values.round(4).to_markdown())
                            
                            # Save as heatmap
                            plt.figure(figsize=(12, 10))
                            mask = np.triu(np.ones_like(p_values, dtype=bool))
                            sns.heatmap(p_values, annot=True, fmt='.3f', cmap='RdYlGn_r',
                                       mask=mask, vmin=0, vmax=0.1, center=0.05,
                                       linewidths=0.5, cbar_kws={'label': 'p-value'})
                            plt.title(f'Nemenyi P-values: Models ({level.title()}, {cv_type.title()}, {metric_label})')
                            plt.tight_layout()
                            plt.savefig(output_dir / f"nemenyi_models_{level}_{cv_type}_{score_type}_{metric_name}.png", 
                                       dpi=300, bbox_inches='tight')
                            plt.close()
                        else:
                            markdown_lines.append("\n*Friedman test not significant, post-hoc not performed.*\n")
                            
                    except Exception as e:
                        markdown_lines.append(f"\n*Error in model comparison: {e}*\n")
                
                # Resampling comparison
                markdown_lines.append("\n### Resampling Comparison\n")
                
                # Create project x resampling matrix (best across models)
                resampling_matrix = df.groupby(['project', 'resampling'])[metric_col].max().unstack()
                resampling_matrix = resampling_matrix.dropna(axis=1, how='all').dropna()
                
                if len(resampling_matrix) >= 3 and len(resampling_matrix.columns) >= 2:
                    try:
                        values = [resampling_matrix[col].values for col in resampling_matrix.columns]
                        stat, p_friedman = friedmanchisquare(*values)
                        k = len(resampling_matrix.columns)  # number of groups
                        df_friedman = k - 1  # degrees of freedom
                        markdown_lines.append(f"**Friedman test:** χ²({df_friedman}) = {stat:.4f}, p = {p_friedman:.6f}\n")
                        
                        if p_friedman < 0.05:
                            p_values = sp.posthoc_nemenyi_friedman(resampling_matrix.values)
                            p_values.index = resampling_matrix.columns
                            p_values.columns = resampling_matrix.columns
                            
                            markdown_lines.append("\n**Nemenyi Post-hoc P-values:**\n")
                            markdown_lines.append(p_values.round(4).to_markdown())
                            
                            # Save as heatmap
                            plt.figure(figsize=(12, 10))
                            mask = np.triu(np.ones_like(p_values, dtype=bool))
                            sns.heatmap(p_values, annot=True, fmt='.3f', cmap='RdYlGn_r',
                                       mask=mask, vmin=0, vmax=0.1, center=0.05,
                                       linewidths=0.5, cbar_kws={'label': 'p-value'})
                            plt.title(f'Nemenyi P-values: Resampling ({level.title()}, {cv_type.title()}, {metric_label})')
                            plt.tight_layout()
                            plt.savefig(output_dir / f"nemenyi_resampling_{level}_{cv_type}_{score_type}_{metric_name}.png", 
                                       dpi=300, bbox_inches='tight')
                            plt.close()
                        else:
                            markdown_lines.append("\n*Friedman test not significant, post-hoc not performed.*\n")
                            
                    except Exception as e:
                        markdown_lines.append(f"\n*Error in resampling comparison: {e}*\n")
            
            with open(output_dir / f"nemenyi_posthoc_{cv_type}_{score_type}_{metric_name}.md", 'w') as f:
                f.write('\n'.join(markdown_lines))
    
    logging.info(f"Nemenyi post-hoc tables saved to {output_dir}")


# =============================================================================
# NEW: Go Metrics Impact Analysis
# =============================================================================

def generate_go_metrics_comparison(output_dir):
    """
    Generate box plot comparing performance with and without Go-specific metrics.
    This shows the impact of Go language features on bug prediction.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    comparison_data = []
    markdown_lines = ["# Go Metrics Impact Analysis\n"]
    markdown_lines.append("Comparing performance with full features vs without Go-specific metrics.\n")
    
    for cv_type in ['temporal', 'shuffle']:
        markdown_lines.append(f"\n## {cv_type.title()} CV\n")
        
        for level in ALL_LEVELS:
            df_full = collect_all_results(level, cv_type, 'full')
            df_no_go = collect_all_results(level, cv_type, 'no_go_metrics')
            
            if df_full.empty or df_no_go.empty:
                continue
            
            # Best per project for each feature set
            best_full = df_full.loc[df_full.groupby('project')['holdout_mcc'].idxmax()]
            best_no_go = df_no_go.loc[df_no_go.groupby('project')['holdout_mcc'].idxmax()]
            
            for _, row in best_full.iterrows():
                comparison_data.append({
                    'Level': level.title(),
                    'CV Type': cv_type.title(),
                    'Feature Set': 'Full (with Go)',
                    'MCC': row['holdout_mcc'],
                    'F1': row['holdout_f1'],
                    'Project': row['project']
                })
            
            for _, row in best_no_go.iterrows():
                comparison_data.append({
                    'Level': level.title(),
                    'CV Type': cv_type.title(),
                    'Feature Set': 'No Go Metrics',
                    'MCC': row['holdout_mcc'],
                    'F1': row['holdout_f1'],
                    'Project': row['project']
                })
            
            # Statistical comparison
            full_scores = best_full.set_index('project')['holdout_mcc']
            no_go_scores = best_no_go.set_index('project')['holdout_mcc']
            
            common = set(full_scores.index) & set(no_go_scores.index)
            if len(common) >= 3:
                full_vals = [full_scores[p] for p in common]
                no_go_vals = [no_go_scores[p] for p in common]
                
                try:
                    stat, p_val = wilcoxon(full_vals, no_go_vals)
                    
                    # Cliff's delta
                    n = len(full_vals)
                    more = sum(1 for f, ng in zip(full_vals, no_go_vals) if f > ng)
                    less = sum(1 for f, ng in zip(full_vals, no_go_vals) if f < ng)
                    delta = (more - less) / (n * n) if n > 0 else 0
                    
                    markdown_lines.append(f"\n### {level.title()} Level\n")
                    markdown_lines.append(f"- Full features mean MCC: {np.mean(full_vals):.4f}")
                    markdown_lines.append(f"- No Go metrics mean MCC: {np.mean(no_go_vals):.4f}")
                    markdown_lines.append(f"- Difference: {np.mean(full_vals) - np.mean(no_go_vals):.4f}")
                    markdown_lines.append(f"- Wilcoxon p-value: {p_val:.4f}")
                    markdown_lines.append(f"- Cliff's δ: {delta:.3f}")
                    markdown_lines.append(f"- Favors: {'Full (with Go)' if delta > 0 else 'No Go Metrics'}")
                except:
                    pass
    
    if comparison_data:
        plot_df = pd.DataFrame(comparison_data)
        
        # Create box plots
        fig, axes = plt.subplots(2, 3, figsize=(16, 10))
        
        for i, cv_type in enumerate(['Temporal', 'Shuffle']):
            cv_data = plot_df[plot_df['CV Type'] == cv_type]
            
            for j, level in enumerate(['Commit', 'File', 'Method']):
                ax = axes[i, j]
                level_data = cv_data[cv_data['Level'] == level]
                
                if not level_data.empty:
                    sns.boxplot(data=level_data, x='Feature Set', y='MCC', ax=ax,
                               palette=['#3498db', '#e74c3c'])
                    sns.stripplot(data=level_data, x='Feature Set', y='MCC', ax=ax,
                                 color='black', alpha=0.5, size=6)
                    
                    ax.set_ylabel('Holdout MCC' if j == 0 else '')
                    ax.set_xlabel('')
                    ax.set_title(f'{level} Level' if i == 0 else '')
                    ax.tick_params(axis='x', rotation=15)
        
        # Add row labels
        axes[0, 0].set_ylabel('Temporal CV\nHoldout MCC')
        axes[1, 0].set_ylabel('Shuffle CV\nHoldout MCC')
        
        plt.suptitle('Impact of Go-Specific Metrics on Bug Prediction Performance', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_dir / "figure_go_metrics_impact.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Also create a simpler combined figure
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        for i, metric in enumerate(['MCC', 'F1']):
            ax = axes[i]
            
            sns.boxplot(data=plot_df, x='Level', y=metric, hue='Feature Set', ax=ax,
                       palette=['#3498db', '#e74c3c'])
            
            ax.set_ylabel(f'Holdout {metric}')
            ax.set_xlabel('Granularity Level')
            ax.set_title(f'{metric} Score')
            ax.legend(title='Feature Set', loc='lower right')
        
        plt.suptitle('Go Metrics Impact: Full vs No-Go Features', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_dir / "figure_go_metrics_impact_simple.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        logging.info(f"Go metrics comparison saved to {output_dir}")
    
    with open(output_dir / "go_metrics_analysis.md", 'w') as f:
        f.write('\n'.join(markdown_lines))


# =============================================================================
# NEW: Comprehensive Model Performance Heatmap
# =============================================================================

def generate_model_performance_heatmap(output_dir, cv_type='temporal', feature_set='full'):
    """
    Generate heatmap showing average performance of each model across all projects.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for level in ALL_LEVELS:
        df = collect_all_results(level, cv_type, feature_set)
        if df.empty:
            continue
        
        # Average MCC per model across projects (best resampling for each project-model)
        model_perf = df.groupby(['project', 'model'])['holdout_mcc'].max().unstack()
        
        if model_perf.empty:
            continue
        
        # Create heatmap
        plt.figure(figsize=(14, 10))
        
        # Sort by mean performance
        model_order = model_perf.mean().sort_values(ascending=False).index
        model_perf = model_perf[model_order]
        
        sns.heatmap(model_perf, annot=True, fmt='.3f', cmap='RdYlGn',
                   center=0.4, linewidths=0.5, cbar_kws={'label': 'Holdout MCC'})
        
        plt.xlabel('Model')
        plt.ylabel('Project')
        plt.title(f'Model Performance Heatmap ({level.title()} Level, {cv_type.title()} CV)')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(output_dir / f"heatmap_model_performance_{level}_{cv_type}.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    logging.info(f"Model performance heatmaps saved to {output_dir}")


# =============================================================================
# NEW: Resampling Performance Heatmap
# =============================================================================

def generate_resampling_performance_heatmap(output_dir, cv_type='temporal', feature_set='full'):
    """
    Generate heatmap showing average performance of each resampling method.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for level in ALL_LEVELS:
        df = collect_all_results(level, cv_type, feature_set)
        if df.empty:
            continue
        
        # Average MCC per resampling across projects (best model for each project-resampling)
        resampling_perf = df.groupby(['project', 'resampling'])['holdout_mcc'].max().unstack()
        
        if resampling_perf.empty:
            continue
        
        # Create heatmap
        plt.figure(figsize=(14, 10))
        
        # Sort by mean performance
        resampling_order = resampling_perf.mean().sort_values(ascending=False).index
        resampling_perf = resampling_perf[resampling_order]
        
        sns.heatmap(resampling_perf, annot=True, fmt='.3f', cmap='RdYlGn',
                   center=0.4, linewidths=0.5, cbar_kws={'label': 'Holdout MCC'})
        
        plt.xlabel('Resampling Method')
        plt.ylabel('Project')
        plt.title(f'Resampling Performance Heatmap ({level.title()} Level, {cv_type.title()} CV)')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(output_dir / f"heatmap_resampling_performance_{level}_{cv_type}.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    logging.info(f"Resampling performance heatmaps saved to {output_dir}")


# =============================================================================
# NEW: Critical Difference Diagram
# =============================================================================

def generate_cd_diagram(output_dir, cv_type='temporal', feature_set='full'):
    """
    Generate Critical Difference diagrams for model and resampling comparison.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for level in ALL_LEVELS:
        df = collect_all_results(level, cv_type, feature_set)
        if df.empty:
            continue
        
        # Model comparison
        model_matrix = df.groupby(['project', 'model'])['holdout_mcc'].max().unstack()
        model_matrix = model_matrix.dropna(axis=1, how='all').dropna()
        
        if len(model_matrix) >= 3 and len(model_matrix.columns) >= 2:
            _draw_cd_diagram(model_matrix, 
                           title=f'Critical Difference: Models ({level.title()} Level)',
                           output_path=output_dir / f"cd_diagram_models_{level}_{cv_type}.png")
        
        # Resampling comparison
        resampling_matrix = df.groupby(['project', 'resampling'])['holdout_mcc'].max().unstack()
        resampling_matrix = resampling_matrix.dropna(axis=1, how='all').dropna()
        
        if len(resampling_matrix) >= 3 and len(resampling_matrix.columns) >= 2:
            _draw_cd_diagram(resampling_matrix,
                           title=f'Critical Difference: Resampling ({level.title()} Level)',
                           output_path=output_dir / f"cd_diagram_resampling_{level}_{cv_type}.png")
    
    logging.info(f"CD diagrams saved to {output_dir}")


def _draw_cd_diagram(data_matrix, title, output_path, alpha=0.05):
    """Draw a Critical Difference diagram."""
    # Calculate average ranks (higher score = lower rank = better)
    ranks = data_matrix.rank(axis=1, ascending=False)
    avg_ranks = ranks.mean().sort_values()
    
    n = len(data_matrix)  # number of observations (projects)
    k = len(data_matrix.columns)  # number of methods
    
    # Critical difference (Nemenyi)
    q_alpha_table = {
        2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728,
        6: 2.850, 7: 2.949, 8: 3.031, 9: 3.102,
        10: 3.164, 11: 3.219, 12: 3.268
    }
    
    q_alpha = q_alpha_table.get(k, 3.3)
    cd = q_alpha * np.sqrt(k * (k + 1) / (6 * n))
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, max(4, k * 0.35)))
    
    # Draw rank axis
    ax.set_xlim(0.5, k + 0.5)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    # Draw top axis line
    ax.axhline(y=0.9, xmin=0.05, xmax=0.95, color='black', linewidth=1.5)
    
    # Draw tick marks and labels
    for i in range(1, k + 1):
        x_pos = 0.05 + (i - 0.5) / k * 0.9
        ax.axvline(x=i, ymin=0.88, ymax=0.92, color='black', linewidth=1)
        ax.text(i, 0.95, str(i), ha='center', va='bottom', fontsize=10)
    
    # Draw CD bar
    ax.plot([1, 1 + cd], [0.83, 0.83], 'k-', linewidth=3)
    ax.plot([1, 1], [0.81, 0.85], 'k-', linewidth=2)
    ax.plot([1 + cd, 1 + cd], [0.81, 0.85], 'k-', linewidth=2)
    ax.text(1 + cd/2, 0.78, f'CD = {cd:.2f}', ha='center', va='top', fontsize=10)
    
    # Position methods
    y_positions = np.linspace(0.7, 0.05, len(avg_ranks))
    
    for i, (method, rank) in enumerate(avg_ranks.items()):
        y = y_positions[i]
        
        # Draw line from method to rank
        ax.plot([rank, rank], [0.88, y + 0.02], 'k-', linewidth=0.7, alpha=0.7)
        ax.scatter([rank], [y + 0.02], color='#2c3e50', s=30, zorder=5)
        
        # Method name with rank
        if rank <= (k + 1) / 2:
            ax.text(rank - 0.15, y, f'{method} ({rank:.2f})', 
                   ha='right', va='center', fontsize=9)
        else:
            ax.text(rank + 0.15, y, f'{method} ({rank:.2f})', 
                   ha='left', va='center', fontsize=9)
    
    # Draw connection bars for non-significant differences
    try:
        p_values = sp.posthoc_nemenyi_friedman(data_matrix.values)
        p_values.index = data_matrix.columns
        p_values.columns = data_matrix.columns
        
        # Find groups that are not significantly different
        connected = []
        sorted_methods = avg_ranks.index.tolist()
        
        for i, m1 in enumerate(sorted_methods):
            for j, m2 in enumerate(sorted_methods[i+1:], i+1):
                if p_values.loc[m1, m2] >= alpha:
                    r1, r2 = avg_ranks[m1], avg_ranks[m2]
                    connected.append((min(r1, r2), max(r1, r2), 0.87 - i * 0.015))
        
        # Draw connection bars
        for r1, r2, y_bar in connected:
            ax.plot([r1, r2], [y_bar, y_bar], 'k-', linewidth=3, alpha=0.6)
    except:
        pass
    
    ax.set_title(title, fontsize=12, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


# =============================================================================
# NEW: Summary Statistics Table
# =============================================================================

def generate_summary_statistics_table(output_dir, cv_type='temporal', feature_set='full'):
    """
    Generate a comprehensive summary statistics table with all metrics.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_stats = []
    
    for level in ALL_LEVELS:
        df = collect_all_results(level, cv_type, feature_set)
        if df.empty:
            continue
        
        # Best per project
        best_per_project = df.loc[df.groupby('project')['holdout_mcc'].idxmax()]
        
        stats_row = {
            'Level': level.title(),
            'n_projects': len(best_per_project),
            'MCC_mean': best_per_project['holdout_mcc'].mean(),
            'MCC_std': best_per_project['holdout_mcc'].std(),
            'MCC_median': best_per_project['holdout_mcc'].median(),
            'MCC_min': best_per_project['holdout_mcc'].min(),
            'MCC_max': best_per_project['holdout_mcc'].max(),
            'F1_mean': best_per_project['holdout_f1'].mean(),
            'F1_std': best_per_project['holdout_f1'].std(),
            'PR_AUC_mean': best_per_project['holdout_pr_auc'].mean(),
            'ROC_AUC_mean': best_per_project['holdout_roc_auc'].mean(),
            'Precision_mean': best_per_project['holdout_precision'].mean(),
            'Recall_mean': best_per_project['holdout_recall'].mean(),
        }
        all_stats.append(stats_row)
    
    stats_df = pd.DataFrame(all_stats)
    
    markdown_lines = [f"# Summary Statistics\n"]
    markdown_lines.append(f"CV Type: {cv_type.title()}, Feature Set: {feature_set}\n")
    markdown_lines.append("\n## Descriptive Statistics by Level\n")
    
    # Formatted table
    markdown_lines.append("| Level | N | MCC (Mean±Std) | MCC Median | MCC Range | F1 (Mean±Std) | PR-AUC | ROC-AUC |")
    markdown_lines.append("|-------|---|----------------|------------|-----------|---------------|--------|---------|")
    
    for _, row in stats_df.iterrows():
        markdown_lines.append(
            f"| {row['Level']} | {row['n_projects']} | "
            f"{row['MCC_mean']:.4f}±{row['MCC_std']:.4f} | {row['MCC_median']:.4f} | "
            f"[{row['MCC_min']:.3f}, {row['MCC_max']:.3f}] | "
            f"{row['F1_mean']:.4f}±{row['F1_std']:.4f} | "
            f"{row['PR_AUC_mean']:.4f} | {row['ROC_AUC_mean']:.4f} |"
        )
    
    with open(output_dir / f"summary_statistics_{cv_type}.md", 'w') as f:
        f.write('\n'.join(markdown_lines))
    
    stats_df.to_csv(output_dir / f"summary_statistics_{cv_type}.csv", index=False)
    
    logging.info(f"Summary statistics saved to {output_dir}")
    
    return stats_df


# =============================================================================
# NEW: CV Score Based Figures (parallel to holdout figures)
# =============================================================================

def generate_cv_score_figures(output_dir, cv_type='temporal', feature_set='full'):
    """
    Generate figures based on CV (cross-validation) scores instead of holdout scores.
    This provides a complementary view to the holdout-based figures.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Level comparison box plots using CV scores
    for metric_name in PRIMARY_METRICS:
        metric_col = f"cv_{metric_name}"
        metric_label = metric_name.upper()
        
        all_data = []
        
        for level in ALL_LEVELS:
            df = collect_all_results(level, cv_type, feature_set)
            if df.empty or metric_col not in df.columns:
                continue
            
            # Best per project (based on same metric in CV)
            df_clean = df.dropna(subset=[metric_col])
            if df_clean.empty:
                continue
                
            best_per_project = df_clean.loc[df_clean.groupby('project')[metric_col].idxmax()]
            
            for _, row in best_per_project.iterrows():
                all_data.append({
                    'Level': level.title(),
                    'Score': row[metric_col],
                    'Project': row['project'],
                    'Metric': metric_label
                })
        
        if not all_data:
            continue
        
        plot_df = pd.DataFrame(all_data)
        
        # Create box plot
        fig, ax = plt.subplots(figsize=(10, 6))
        colors = {'Commit': '#e74c3c', 'File': '#3498db', 'Method': '#2ecc71'}
        
        sns.boxplot(data=plot_df, x='Level', y='Score', ax=ax,
                   palette=colors, order=['Commit', 'File', 'Method'])
        sns.stripplot(data=plot_df, x='Level', y='Score', ax=ax,
                     color='black', alpha=0.5, size=6, order=['Commit', 'File', 'Method'])
        
        ax.set_ylabel(f'CV {metric_label}')
        ax.set_xlabel('Granularity Level')
        ax.set_title(f'Level Comparison by CV {metric_label} ({cv_type.title()} CV)')
        
        # Add Friedman test
        level_scores = {}
        for level in ALL_LEVELS:
            level_df = plot_df[plot_df['Level'] == level.title()]
            if not level_df.empty:
                level_scores[level] = level_df.set_index('Project')['Score']
        
        if len(level_scores) == 3:
            combined = pd.DataFrame(level_scores).dropna()
            if len(combined) >= 3:
                try:
                    stat, p_val = friedmanchisquare(
                        combined['commit'].values,
                        combined['file'].values,
                        combined['method'].values
                    )
                    k = len(combined.columns)
                    df_friedman = k - 1
                    sig_text = f'Friedman: χ²({df_friedman})={stat:.4f}, p={p_val:.4f}'
                    if p_val < 0.05:
                        sig_text += ' *'
                    ax.text(0.02, 0.98, sig_text, transform=ax.transAxes,
                           fontsize=10, va='top', ha='left',
                           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
                except:
                    pass
        
        plt.tight_layout()
        plt.savefig(output_dir / f"figure_level_comparison_cv_{metric_name}_{cv_type}.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    # 2. Combined CV metrics figure (all metrics in one figure)
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()
    
    cv_metrics_to_plot = ['cv_mcc', 'cv_f1', 'cv_pr_auc', 'cv_roc_auc']
    
    for i, metric_col in enumerate(cv_metrics_to_plot):
        ax = axes[i]
        metric_label = METRIC_SHORT_LABELS.get(metric_col, metric_col)
        
        all_data = []
        for level in ALL_LEVELS:
            df = collect_all_results(level, cv_type, feature_set)
            if df.empty or metric_col not in df.columns:
                continue
            
            df_clean = df.dropna(subset=[metric_col])
            if df_clean.empty:
                continue
                
            best_per_project = df_clean.loc[df_clean.groupby('project')[metric_col].idxmax()]
            
            for _, row in best_per_project.iterrows():
                all_data.append({
                    'Level': level.title(),
                    'Score': row[metric_col],
                    'Project': row['project']
                })
        
        if all_data:
            plot_df = pd.DataFrame(all_data)
            colors = {'Commit': '#e74c3c', 'File': '#3498db', 'Method': '#2ecc71'}
            
            sns.boxplot(data=plot_df, x='Level', y='Score', ax=ax,
                       palette=colors, order=['Commit', 'File', 'Method'])
            sns.stripplot(data=plot_df, x='Level', y='Score', ax=ax,
                         color='black', alpha=0.5, size=5, order=['Commit', 'File', 'Method'])
            
            ax.set_ylabel(f'CV {metric_label}')
            ax.set_xlabel('')
            ax.set_title(f'{metric_label}')
    
    plt.suptitle(f'Level Comparison by CV Scores ({cv_type.title()} CV)', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / f"figure_level_comparison_cv_all_metrics_{cv_type}.png", 
               dpi=300, bbox_inches='tight')
    plt.close()
    
    logging.info(f"CV score figures saved to {output_dir}")


def generate_holdout_vs_cv_comparison(output_dir, cv_type='temporal', feature_set='full'):
    """
    Generate figures comparing holdout scores vs CV scores.
    This shows how well CV performance predicts holdout performance.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for i, level in enumerate(ALL_LEVELS):
        ax = axes[i]
        df = collect_all_results(level, cv_type, feature_set)
        
        if df.empty:
            continue
        
        # Get best configuration per project based on CV MCC
        df_clean = df.dropna(subset=['cv_mcc', 'holdout_mcc'])
        if df_clean.empty:
            continue
        
        best_per_project = df_clean.loc[df_clean.groupby('project')['cv_mcc'].idxmax()]
        
        cv_scores = best_per_project['cv_mcc'].values
        holdout_scores = best_per_project['holdout_mcc'].values
        
        ax.scatter(cv_scores, holdout_scores, alpha=0.7, s=80, c='steelblue')
        
        # Add diagonal line
        min_val = min(min(cv_scores), min(holdout_scores))
        max_val = max(max(cv_scores), max(holdout_scores))
        ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, label='Perfect correlation')
        
        # Add project labels
        for _, row in best_per_project.iterrows():
            ax.annotate(row['project'], (row['cv_mcc'], row['holdout_mcc']), 
                       fontsize=7, alpha=0.7)
        
        # Calculate correlation
        from scipy.stats import pearsonr, spearmanr
        r_pearson, p_pearson = pearsonr(cv_scores, holdout_scores)
        r_spearman, p_spearman = spearmanr(cv_scores, holdout_scores)
        
        ax.text(0.05, 0.95, f'Pearson r={r_pearson:.3f}\nSpearman ρ={r_spearman:.3f}',
               transform=ax.transAxes, fontsize=9, va='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        ax.set_xlabel('CV MCC')
        ax.set_ylabel('Holdout MCC')
        ax.set_title(f'{level.title()} Level')
    
    plt.suptitle(f'CV vs Holdout Performance Correlation ({cv_type.title()} CV)', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / f"figure_cv_vs_holdout_correlation_{cv_type}.png", 
               dpi=300, bbox_inches='tight')
    plt.close()
    
    logging.info(f"CV vs Holdout comparison saved to {output_dir}")


def generate_model_heatmap_cv_scores(output_dir, cv_type='temporal', feature_set='full'):
    """
    Generate model performance heatmaps using CV scores.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for metric_name in PRIMARY_METRICS:
        metric_col = f"cv_{metric_name}"
        metric_label = f"CV {metric_name.upper()}"
        
        for level in ALL_LEVELS:
            df = collect_all_results(level, cv_type, feature_set)
            if df.empty or metric_col not in df.columns:
                continue
            
            df_clean = df.dropna(subset=[metric_col])
            if df_clean.empty:
                continue
            
            # Average score per model across projects (best resampling for each project-model)
            model_perf = df_clean.groupby(['project', 'model'])[metric_col].max().unstack()
            
            if model_perf.empty:
                continue
            
            # Create heatmap
            plt.figure(figsize=(14, 10))
            
            # Sort by mean performance
            model_order = model_perf.mean().sort_values(ascending=False).index
            model_perf = model_perf[model_order]
            
            sns.heatmap(model_perf, annot=True, fmt='.3f', cmap='RdYlGn',
                       center=0.4, linewidths=0.5, cbar_kws={'label': metric_label})
            
            plt.xlabel('Model')
            plt.ylabel('Project')
            plt.title(f'Model Performance: {metric_label} ({level.title()} Level, {cv_type.title()} CV)')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(output_dir / f"heatmap_model_cv_{metric_name}_{level}_{cv_type}.png", 
                       dpi=300, bbox_inches='tight')
            plt.close()
    
    logging.info(f"Model CV heatmaps saved to {output_dir}")


def generate_resampling_heatmap_cv_scores(output_dir, cv_type='temporal', feature_set='full'):
    """
    Generate resampling performance heatmaps using CV scores.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for metric_name in PRIMARY_METRICS:
        metric_col = f"cv_{metric_name}"
        metric_label = f"CV {metric_name.upper()}"
        
        for level in ALL_LEVELS:
            df = collect_all_results(level, cv_type, feature_set)
            if df.empty or metric_col not in df.columns:
                continue
            
            df_clean = df.dropna(subset=[metric_col])
            if df_clean.empty:
                continue
            
            # Average score per resampling across projects
            resampling_perf = df_clean.groupby(['project', 'resampling'])[metric_col].max().unstack()
            
            if resampling_perf.empty:
                continue
            
            plt.figure(figsize=(14, 10))
            
            resampling_order = resampling_perf.mean().sort_values(ascending=False).index
            resampling_perf = resampling_perf[resampling_order]
            
            sns.heatmap(resampling_perf, annot=True, fmt='.3f', cmap='RdYlGn',
                       center=0.4, linewidths=0.5, cbar_kws={'label': metric_label})
            
            plt.xlabel('Resampling Method')
            plt.ylabel('Project')
            plt.title(f'Resampling Performance: {metric_label} ({level.title()} Level, {cv_type.title()} CV)')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(output_dir / f"heatmap_resampling_cv_{metric_name}_{level}_{cv_type}.png", 
                       dpi=300, bbox_inches='tight')
            plt.close()
    
    logging.info(f"Resampling CV heatmaps saved to {output_dir}")


def generate_cd_diagram_multi_metric(output_dir, cv_type='temporal', feature_set='full'):
    """
    Generate Critical Difference diagrams for multiple metrics (MCC, F1) and score types (holdout, CV).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for score_type in SCORE_TYPES:
        for metric_name in PRIMARY_METRICS:
            metric_col = f"{score_type}_{metric_name}"
            metric_label = f"{metric_name.upper()} ({score_type.title()})"
            
            for level in ALL_LEVELS:
                df = collect_all_results(level, cv_type, feature_set)
                if df.empty or metric_col not in df.columns:
                    continue
                
                df_clean = df.dropna(subset=[metric_col])
                if df_clean.empty:
                    continue
                
                # Model comparison
                model_matrix = df_clean.groupby(['project', 'model'])[metric_col].max().unstack()
                model_matrix = model_matrix.dropna(axis=1, how='all').dropna()
                
                if len(model_matrix) >= 3 and len(model_matrix.columns) >= 2:
                    _draw_cd_diagram(model_matrix, 
                                   title=f'CD: Models ({level.title()}, {metric_label})',
                                   output_path=output_dir / f"cd_models_{level}_{cv_type}_{score_type}_{metric_name}.png")
                
                # Resampling comparison
                resampling_matrix = df_clean.groupby(['project', 'resampling'])[metric_col].max().unstack()
                resampling_matrix = resampling_matrix.dropna(axis=1, how='all').dropna()
                
                if len(resampling_matrix) >= 3 and len(resampling_matrix.columns) >= 2:
                    _draw_cd_diagram(resampling_matrix,
                                   title=f'CD: Resampling ({level.title()}, {metric_label})',
                                   output_path=output_dir / f"cd_resampling_{level}_{cv_type}_{score_type}_{metric_name}.png")
    
    logging.info(f"Multi-metric CD diagrams saved to {output_dir}")


def generate_all_tables_and_figures(output_dir=None):
    """Generate all academic tables and figures."""
    if output_dir is None:
        output_dir = BASE_DIR / "academic_outputs"
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logging.info("=" * 60)
    logging.info("Generating Academic Tables and Figures")
    logging.info("=" * 60)
    
    total_steps = 18
    
    # Table 1: Dataset Statistics
    logging.info(f"\n[1/{total_steps}] Generating Dataset Statistics Table...")
    generate_dataset_statistics_table(output_dir)
    
    # Table 2: Level Comparison (default temporal)
    logging.info(f"\n[2/{total_steps}] Generating Level Comparison Table (Temporal)...")
    generate_level_comparison_table(output_dir, cv_type='temporal')
    
    # Table 2b: Level Comparison (shuffle)
    logging.info(f"\n[3/{total_steps}] Generating Level Comparison Table (Shuffle)...")
    generate_level_comparison_table(output_dir, cv_type='shuffle')
    
    # Table 3: CV Comparison
    logging.info(f"\n[4/{total_steps}] Generating CV Comparison Table...")
    generate_cv_comparison(output_dir)
    
    # Table 4: Best Configurations
    logging.info(f"\n[5/{total_steps}] Generating Best Configuration Table...")
    generate_best_configuration_table(output_dir)
    
    # Table 5: Winning Frequency
    logging.info(f"\n[6/{total_steps}] Generating Winning Frequency Table...")
    generate_model_resampling_frequency_table(output_dir)
    
    # Radar Chart
    logging.info(f"\n[7/{total_steps}] Generating Radar Charts...")
    generate_radar_chart(output_dir, cv_type='temporal')
    generate_radar_chart(output_dir, cv_type='shuffle')
    
    # Level Comparison by Metric (temporal and shuffle) - HOLDOUT
    logging.info(f"\n[8/{total_steps}] Generating Level Comparison by Metric (Holdout)...")
    generate_level_comparison_by_metric(output_dir)
    
    # Nemenyi Post-hoc Tables (now with MCC and F1, holdout and CV)
    logging.info(f"\n[9/{total_steps}] Generating Nemenyi Post-hoc Tables (MCC & F1, Holdout & CV)...")
    generate_nemenyi_tables(output_dir, cv_type='temporal')
    generate_nemenyi_tables(output_dir, cv_type='shuffle')
    
    # Go Metrics Impact Analysis
    logging.info(f"\n[10/{total_steps}] Generating Go Metrics Impact Analysis...")
    generate_go_metrics_comparison(output_dir)
    
    # Model Performance Heatmaps (Holdout)
    logging.info(f"\n[11/{total_steps}] Generating Model Performance Heatmaps (Holdout)...")
    generate_model_performance_heatmap(output_dir, cv_type='temporal')
    generate_model_performance_heatmap(output_dir, cv_type='shuffle')
    
    # Resampling Performance Heatmaps (Holdout)
    logging.info(f"\n[12/{total_steps}] Generating Resampling Performance Heatmaps (Holdout)...")
    generate_resampling_performance_heatmap(output_dir, cv_type='temporal')
    generate_resampling_performance_heatmap(output_dir, cv_type='shuffle')
    
    # Critical Difference Diagrams (Multi-metric: MCC & F1, Holdout & CV)
    logging.info(f"\n[13/{total_steps}] Generating CD Diagrams (MCC & F1, Holdout & CV)...")
    generate_cd_diagram_multi_metric(output_dir, cv_type='temporal')
    generate_cd_diagram_multi_metric(output_dir, cv_type='shuffle')
    
    # Summary Statistics
    logging.info(f"\n[14/{total_steps}] Generating Summary Statistics...")
    generate_summary_statistics_table(output_dir, cv_type='temporal')
    generate_summary_statistics_table(output_dir, cv_type='shuffle')
    
    # NEW: CV Score Figures
    logging.info(f"\n[15/{total_steps}] Generating CV Score Figures...")
    generate_cv_score_figures(output_dir, cv_type='temporal')
    generate_cv_score_figures(output_dir, cv_type='shuffle')
    
    # NEW: Holdout vs CV Correlation
    logging.info(f"\n[16/{total_steps}] Generating Holdout vs CV Correlation...")
    generate_holdout_vs_cv_comparison(output_dir, cv_type='temporal')
    generate_holdout_vs_cv_comparison(output_dir, cv_type='shuffle')
    
    # NEW: Model Heatmaps with CV Scores
    logging.info(f"\n[17/{total_steps}] Generating Model Heatmaps (CV Scores)...")
    generate_model_heatmap_cv_scores(output_dir, cv_type='temporal')
    generate_model_heatmap_cv_scores(output_dir, cv_type='shuffle')
    
    # NEW: Resampling Heatmaps with CV Scores
    logging.info(f"\n[18/{total_steps}] Generating Resampling Heatmaps (CV Scores)...")
    generate_resampling_heatmap_cv_scores(output_dir, cv_type='temporal')
    generate_resampling_heatmap_cv_scores(output_dir, cv_type='shuffle')
    
    logging.info("\n" + "=" * 60)
    logging.info(f"All outputs saved to: {output_dir}")
    logging.info("=" * 60)
    
    # List generated files
    print("\nGenerated Files:")
    
    # Tables
    print("\n📋 Tables (Markdown):")
    for f in sorted(output_dir.glob("*.md")):
        print(f"  - {f.name}")
    
    # Figures
    print("\n📊 Figures (PNG):")
    png_files = sorted(output_dir.glob("*.png"))
    print(f"  Total: {len(png_files)} files")
    # Group by prefix
    prefixes = set(f.name.split('_')[0] for f in png_files)
    for prefix in sorted(prefixes):
        count = sum(1 for f in png_files if f.name.startswith(prefix))
        print(f"    {prefix}_*: {count} files")
    
    # Data files
    print("\n📁 Data (CSV):")
    for f in sorted(output_dir.glob("*.csv")):
        print(f"  - {f.name}")


def main():
    parser = argparse.ArgumentParser(description='Generate Academic Figures and Tables')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory for generated files')
    parser.add_argument('--cv-type', type=str, default='temporal',
                        choices=['temporal', 'shuffle'],
                        help='CV type to analyze')
    parser.add_argument('--feature-set', type=str, default='full',
                        choices=['full', 'no_go_metrics'],
                        help='Feature set to analyze')
    parser.add_argument('--table', type=str, default='all',
                        choices=['all', 'dataset', 'level', 'cv', 'best', 'frequency', 
                                'radar', 'level_metrics', 'nemenyi', 'go_metrics',
                                'model_heatmap', 'resampling_heatmap', 'cd_diagram', 
                                'summary_stats', 'cv_scores', 'cv_vs_holdout',
                                'model_heatmap_cv', 'resampling_heatmap_cv', 'cd_multi_metric'],
                        help='Which table/figure to generate')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir) if args.output_dir else BASE_DIR / "academic_outputs"
    
    if args.table == 'all':
        generate_all_tables_and_figures(output_dir)
    elif args.table == 'dataset':
        generate_dataset_statistics_table(output_dir)
    elif args.table == 'level':
        generate_level_comparison_table(output_dir, args.cv_type, args.feature_set)
    elif args.table == 'cv':
        generate_cv_comparison(output_dir, args.feature_set)
    elif args.table == 'best':
        generate_best_configuration_table(output_dir, args.cv_type, args.feature_set)
    elif args.table == 'frequency':
        generate_model_resampling_frequency_table(output_dir, args.cv_type, args.feature_set)
    elif args.table == 'radar':
        generate_radar_chart(output_dir, args.cv_type, args.feature_set)
    elif args.table == 'level_metrics':
        generate_level_comparison_by_metric(output_dir)
    elif args.table == 'nemenyi':
        generate_nemenyi_tables(output_dir, args.cv_type, args.feature_set)
    elif args.table == 'go_metrics':
        generate_go_metrics_comparison(output_dir)
    elif args.table == 'model_heatmap':
        generate_model_performance_heatmap(output_dir, args.cv_type, args.feature_set)
    elif args.table == 'resampling_heatmap':
        generate_resampling_performance_heatmap(output_dir, args.cv_type, args.feature_set)
    elif args.table == 'cd_diagram':
        generate_cd_diagram(output_dir, args.cv_type, args.feature_set)
    elif args.table == 'summary_stats':
        generate_summary_statistics_table(output_dir, args.cv_type, args.feature_set)
    elif args.table == 'cv_scores':
        generate_cv_score_figures(output_dir, args.cv_type, args.feature_set)
    elif args.table == 'cv_vs_holdout':
        generate_holdout_vs_cv_comparison(output_dir, args.cv_type, args.feature_set)
    elif args.table == 'model_heatmap_cv':
        generate_model_heatmap_cv_scores(output_dir, args.cv_type, args.feature_set)
    elif args.table == 'resampling_heatmap_cv':
        generate_resampling_heatmap_cv_scores(output_dir, args.cv_type, args.feature_set)
    elif args.table == 'cd_multi_metric':
        generate_cd_diagram_multi_metric(output_dir, args.cv_type, args.feature_set)


if __name__ == '__main__':
    main()
