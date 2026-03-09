#!/usr/bin/env python3
"""
Generate Top 5 Most Influential Features Tables for Each Analysis Level.

This script aggregates feature_scores from all analysis_summary.json files
across projects and generates per-level "Top 5 Most Influential Features"
tables for academic publication.

Methodology:
    1. For each level (commit, file, method), collect feature_scores from
       all analysis_summary.json files that used the 'combine' feature
       selection method with 'full' feature set.
    2. Across all projects, normalize each project's scores to [0, 1] range,
       then compute the mean normalized importance per feature.
    3. Rank features by mean normalized importance and report Top 5.
    4. Also report the number of projects where each feature appeared,
       showing cross-project consistency.

Output:
    - academic_outputs/table_top_features_commit.md
    - academic_outputs/table_top_features_file.md
    - academic_outputs/table_top_features_method.md
    - academic_outputs/table_top_features_all_levels.md  (combined table)
"""

import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from datetime import datetime


LEVELS = ["commit", "file", "method"]
RESULTS_DIRS = {
    "commit": "results_commit_level",
    "file": "results_file_level",
    "method": "results_method_level",
}
OUTPUT_DIR = "academic_outputs"

# We aggregate from experiments that use:
#   - 'combine' feature selection (ensemble of 6 methods)
#   - 'full' feature set (not no_go_metrics)
# We take both CV types (temporal, shuffle) into account
TARGET_FEATURE_SET = "full"


def collect_feature_scores(level):
    """
    Collect feature_scores from all analysis_summary.json files for a level.

    Returns a dict: {project_name: {feature: score, ...}, ...}
    Only includes runs with 'combine' feature selection and 'full' feature set.
    If a project has multiple runs (different CV types / resampling), we average
    scores across those runs per project to get one score vector per project.
    """
    results_dir = Path(RESULTS_DIRS[level])
    if not results_dir.exists():
        print(f"  Warning: {results_dir} does not exist, skipping.")
        return {}

    # Collect all feature score dicts per project
    # Structure: {project: [{ feature: score, ... }, ...]}
    project_scores = defaultdict(list)

    for summary_file in sorted(results_dir.rglob("analysis_summary.json")):
        # Parse path to get project and feature_set
        # Expected path: results_{level}_level/{project}/{cv_type}/{feature_set}/{resampling}/analysis_summary.json
        parts = summary_file.relative_to(results_dir).parts
        if len(parts) < 4:
            continue

        project = parts[0]
        # cv_type = parts[1]
        feature_set = parts[2]

        # Only include 'full' feature set
        if feature_set != TARGET_FEATURE_SET:
            continue

        try:
            with open(summary_file, "r") as f:
                data = json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            print(f"  Warning: Could not read {summary_file}: {e}")
            continue

        fs = data.get("feature_selection", {})
        if not fs:
            continue

        # Only include 'combine' method
        if fs.get("method") != "combine":
            continue

        scores = fs.get("feature_scores", {})
        if not scores:
            continue

        # Filter out NaN values
        valid_scores = {}
        for feat, score in scores.items():
            if score is not None and not (isinstance(score, float) and (score != score)):  # NaN check
                valid_scores[feat] = score

        if valid_scores:
            project_scores[project].append(valid_scores)

    # Average scores per project (across different CV types / resampling combos)
    project_avg_scores = {}
    for project, score_list in project_scores.items():
        feature_sums = defaultdict(float)
        feature_counts = defaultdict(int)

        for scores in score_list:
            for feat, score in scores.items():
                feature_sums[feat] += score
                feature_counts[feat] += 1

        project_avg_scores[project] = {
            feat: feature_sums[feat] / feature_counts[feat]
            for feat in feature_sums
        }

    return project_avg_scores


def aggregate_across_projects(project_avg_scores):
    """
    Aggregate feature importance scores across all projects.

    For each project, normalize scores to [0, 1] (dividing by max score in that project),
    then compute the mean normalized importance across projects.

    Returns a sorted list of (feature, mean_normalized_score, n_projects_appeared).
    """
    if not project_avg_scores:
        return []

    # Normalize each project's scores to [0, 1]
    normalized_scores = defaultdict(list)

    for project, scores in project_avg_scores.items():
        if not scores:
            continue

        max_score = max(abs(v) for v in scores.values())
        if max_score == 0:
            continue

        for feat, score in scores.items():
            normalized_scores[feat].append(abs(score) / max_score)

    # Compute mean normalized importance and count
    results = []
    total_projects = len(project_avg_scores)
    for feat, scores_list in normalized_scores.items():
        mean_score = sum(scores_list) / len(scores_list)
        n_projects = len(scores_list)
        results.append((feat, mean_score, n_projects, total_projects))

    # Sort by mean normalized importance descending
    results.sort(key=lambda x: x[1], reverse=True)
    return results


def format_feature_name(feature):
    """Format feature name for display in academic tables."""
    # Convert snake_case to readable but keep it technical
    return f"`{feature}`"


def generate_level_table(level, ranked_features, top_n=5):
    """Generate a markdown table for one level."""
    level_display = level.capitalize()

    lines = [
        f"# Top {top_n} Most Influential Features at the {level_display} Level",
        "",
        f"Features ranked by mean normalized importance across all projects.",
        f"Scores are computed using an ensemble of 6 feature selection methods",
        f"(Variance Threshold, Chi-Square, RFE, LASSO, Random Forest, Mutual Information),",
        f"normalized per-method to [0, 1], and averaged across projects.",
        "",
    ]

    if not ranked_features:
        lines.append("No feature importance data available for this level.")
        lines.append("")
        return "\n".join(lines)

    # Top N table
    lines.append(f"| Rank | Feature | Mean Normalized Importance | Projects (n) |")
    lines.append(f"|:----:|:--------|:-------------------------:|:------------:|")

    for i, (feat, score, n_proj, total_proj) in enumerate(ranked_features[:top_n]):
        lines.append(
            f"| {i + 1} | {format_feature_name(feat)} | {score:.4f} | {n_proj}/{total_proj} |"
        )

    lines.append("")

    # Full ranking (collapsed for reference)
    lines.append("<details>")
    lines.append(f"<summary>Full Feature Ranking ({len(ranked_features)} features)</summary>")
    lines.append("")
    lines.append(f"| Rank | Feature | Mean Normalized Importance | Projects (n) |")
    lines.append(f"|:----:|:--------|:-------------------------:|:------------:|")

    for i, (feat, score, n_proj, total_proj) in enumerate(ranked_features):
        lines.append(
            f"| {i + 1} | {format_feature_name(feat)} | {score:.4f} | {n_proj}/{total_proj} |"
        )

    lines.append("")
    lines.append("</details>")
    lines.append("")
    lines.append(f"*Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")

    return "\n".join(lines)


def generate_combined_table(all_results, top_n=5):
    """Generate a combined table showing Top 5 for all levels side by side."""
    lines = [
        "# Top 5 Most Influential Features by Analysis Level",
        "",
        "Features ranked by mean normalized importance (ensemble of 6 feature selection methods)",
        "across all projects per level.",
        "",
    ]

    for level in LEVELS:
        level_display = level.capitalize()
        ranked = all_results.get(level, [])

        lines.append(f"## {level_display} Level")
        lines.append("")

        if not ranked:
            lines.append("No feature importance data available.")
            lines.append("")
            continue

        lines.append(f"| Rank | Feature | Mean Norm. Importance | Projects |")
        lines.append(f"|:----:|:--------|:--------------------:|:--------:|")

        for i, (feat, score, n_proj, total_proj) in enumerate(ranked[:top_n]):
            lines.append(
                f"| {i + 1} | {format_feature_name(feat)} | {score:.4f} | {n_proj}/{total_proj} |"
            )

        lines.append("")

    lines.append(f"*Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")

    return "\n".join(lines)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    all_results = {}
    top_n = 5

    print("=" * 70)
    print("  Generating Top Feature Importance Tables")
    print("=" * 70)

    for level in LEVELS:
        print(f"\nProcessing {level} level...")

        # Step 1: Collect feature scores from all projects
        project_scores = collect_feature_scores(level)
        print(f"  Found data for {len(project_scores)} projects: {', '.join(sorted(project_scores.keys()))}")

        # Step 2: Aggregate across projects
        ranked_features = aggregate_across_projects(project_scores)
        all_results[level] = ranked_features

        if ranked_features:
            print(f"  Total unique features: {len(ranked_features)}")
            print(f"  Top {top_n} features:")
            for i, (feat, score, n_proj, total_proj) in enumerate(ranked_features[:top_n]):
                print(f"    {i + 1}. {feat}: {score:.4f} ({n_proj}/{total_proj} projects)")
        else:
            print("  No feature importance data found.")

        # Step 3: Generate per-level markdown table
        md_content = generate_level_table(level, ranked_features, top_n)
        output_path = os.path.join(OUTPUT_DIR, f"table_top_features_{level}.md")
        with open(output_path, "w") as f:
            f.write(md_content)
        print(f"  Saved: {output_path}")

    # Step 4: Generate combined table
    combined_content = generate_combined_table(all_results, top_n)
    combined_path = os.path.join(OUTPUT_DIR, "table_top_features_all_levels.md")
    with open(combined_path, "w") as f:
        f.write(combined_content)
    print(f"\nSaved combined table: {combined_path}")

    print("\n" + "=" * 70)
    print("  Feature importance table generation complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
