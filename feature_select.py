import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path
import argparse
from sklearn.feature_selection import (
    VarianceThreshold, SelectKBest, chi2, RFE, SelectFromModel,
    mutual_info_classif
)
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def select_features(X, y, method='variance', k=None):
    """
    Apply feature selection using the specified method

    Parameters:
    -----------
    X : DataFrame
        Features
    y : Series
        Target variable
    method : str
        Feature selection method ('variance', 'chi2', 'rfe', 'lasso', 'rf', 'mi')
    k : int
        Number of features to select (if applicable)

    Returns:
    --------
    X_selected : DataFrame
        Selected features
    selected_features : list
        Names of selected features
    importance_scores : dict
        Feature importance scores if available
    """
    if X.shape[1] == 0:
        return X, [], {}

    if k is None:
        k = max(1, X.shape[1] // 2)  # Default to selecting half of features, minimum 1

    k = min(k, X.shape[1])  # Make sure k is not larger than number of features

    importance_scores = {}

    try:
        if method == 'variance':
            # Variance Threshold should be applied on the original, unscaled data.
            selector = VarianceThreshold(threshold=0.01)
            X_selected = selector.fit_transform(X)
            selected_features = X.columns[selector.get_support()].tolist()
            importance_scores = {feat: var for feat, var in zip(X.columns, selector.variances_)}

        elif method == 'chi2':
            # Ensure features are non-negative for chi-square test by using MinMaxScaler on original data.
            min_max_scaler = MinMaxScaler()
            X_scaled = min_max_scaler.fit_transform(X)
            X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)

            selector = SelectKBest(chi2, k=k)
            X_selected = selector.fit_transform(X_scaled_df, y)
            selected_features = X.columns[selector.get_support()].tolist()
            importance_scores = {feat: score for feat, score in zip(X.columns, selector.scores_)}

        else: # All other methods can use StandardScaler
            # Create a copy for scaling
            X_for_selection = X.copy()
            # Identify boolean columns
            bool_cols = [col for col in X.columns if X[col].dtype == bool or set(X[col].unique()).issubset({0, 1, True, False})]
            # For non-boolean columns, apply scaling
            non_bool_cols = [col for col in X.columns if col not in bool_cols]

            if non_bool_cols:
                try:
                    scaler = StandardScaler()
                    X_scaled = scaler.fit_transform(X[non_bool_cols])
                    X_for_selection[non_bool_cols] = X_scaled
                except Exception as e:
                    print(f"  Warning: Scaling error: {str(e)}. Using unscaled features.")

            if method == 'rfe':
                # Recursive Feature Elimination with Random Forest
                estimator = RandomForestClassifier(n_estimators=100, random_state=42)
                selector = RFE(estimator=estimator, n_features_to_select=k)
                X_selected = selector.fit_transform(X_for_selection, y)
                selected_features = X.columns[selector.get_support()].tolist()
                importance_scores = {feat: rank for feat, rank in zip(X.columns, selector.ranking_)}

            elif method == 'lasso':
                # LASSO Feature Selection
                selector = SelectFromModel(Lasso(alpha=0.01, random_state=42), max_features=k)
                X_selected = selector.fit_transform(X_for_selection, y)
                selected_features = X.columns[selector.get_support()].tolist()
                importance_scores = {feat: abs(imp) for feat, imp in zip(X.columns, selector.estimator_.coef_)}

            elif method == 'rf':
                # Random Forest Feature Importance
                selector = SelectFromModel(
                    RandomForestClassifier(n_estimators=100, random_state=42),
                    max_features=k
                )
                X_selected = selector.fit_transform(X_for_selection, y)
                selected_features = X.columns[selector.get_support()].tolist()
                importance_scores = {feat: imp for feat, imp in zip(X.columns, selector.estimator_.feature_importances_)}

            elif method == 'mi':
                # Mutual Information
                selector = SelectKBest(mutual_info_classif, k=k)
                X_selected = selector.fit_transform(X_for_selection, y)
                selected_features = X.columns[selector.get_support()].tolist()
                importance_scores = {feat: score for feat, score in zip(X.columns, selector.scores_)}

            else:
                raise ValueError(f"Unknown feature selection method: {method}")

    except Exception as e:
        print(f"  Error in {method} feature selection: {str(e)}")
        # Fallback: return all features if the selection method fails
        return X, X.columns.tolist(), {feat: 1.0 for feat in X.columns}

    return X[selected_features], selected_features, importance_scores

def analyze_feature_selection_results(df, selected_features, importance_scores, method, output_dir):
    """Analyze and visualize feature selection results"""
    # Create output directory for feature selection results
    fs_dir = os.path.join(output_dir, 'feature_selection')
    os.makedirs(fs_dir, exist_ok=True)

    # Save selected features list with importance scores
    with open(os.path.join(fs_dir, f'{method}_selected_features.txt'), 'w') as f:
        f.write("Selected Features (with importance scores):\n")
        f.write("-" * 50 + "\n")
        for feature in selected_features:
            score = importance_scores.get(feature, 'N/A')
            f.write(f"{feature}: {score}\n")

    try:
        # Plot feature importance visualization (only if we have numeric scores)
        if all(isinstance(score, (int, float)) for score in importance_scores.values()):
            plt.figure(figsize=(12, 8))
            importance_df = pd.Series(importance_scores)
            importance_df = importance_df.sort_values(ascending=True)

            # Limit to top 30 features for readability if there are many
            if len(importance_df) > 30:
                importance_df = importance_df.iloc[-30:]

            importance_df.plot(kind='barh')
            plt.title(f'Feature Importance Scores ({method.upper()})')
            plt.xlabel('Importance Score')
            plt.tight_layout()
            plt.savefig(os.path.join(fs_dir, f'{method}_feature_importance.png'))
            plt.close()
    except Exception as e:
        print(f"  Warning: Error generating importance plot: {str(e)}")

    try:
        # Calculate and save correlation matrix for selected features
        # Only include numeric features (no boolean) to avoid correlation issues
        numeric_features = []
        for feature in selected_features:
            # Check if column is in df (in case features were transformed)
            if feature in df.columns:
                if df[feature].dtype != bool and not set(df[feature].unique()).issubset({0, 1, True, False}):
                    numeric_features.append(feature)

        # Only create correlation matrix if we have at least 2 numeric features
        if len(numeric_features) >= 2:
            corr = df[numeric_features].corr()

            # Limit size of correlation matrix visualization
            plt.figure(figsize=(min(12, len(numeric_features) * 0.5 + 2),
                               min(10, len(numeric_features) * 0.5 + 2)))

            # For large matrices, don't show annotations
            annot = len(numeric_features) <= 15
            sns.heatmap(corr, annot=annot, cmap='coolwarm', center=0,
                        linewidths=.5, square=True)
            plt.title(f'Correlation Matrix ({method.upper()})')
            plt.tight_layout()
            plt.savefig(os.path.join(fs_dir, f'{method}_correlation_matrix.png'))
            plt.close()
    except Exception as e:
        print(f"  Warning: Error generating correlation matrix: {str(e)}")

def combine_feature_importance(all_results):
    """
    Combine and analyze results from all feature selection methods.
    
    Features are penalized if they:
    - Have NaN scores (e.g., zero variance for chi2)
    - Have zero importance across multiple methods
    
    The final score considers both the average normalized importance
    and the consistency across methods.
    """
    combined_scores = {}
    method_count = len(all_results)

    # Normalize scores for each method and combine them
    for method, (_, selected_feats, importance) in all_results.items():
        if importance:
            # Filter out NaN values before finding max
            valid_scores = {k: v for k, v in importance.items() 
                          if v is not None and not (isinstance(v, float) and np.isnan(v))}
            
            if not valid_scores:
                continue
                
            max_score = max(abs(score) for score in valid_scores.values())
            if max_score == 0:
                continue  # Skip if all scores are zero
                
            # Normalize scores to 0-1 range, treating NaN as 0
            for feat, score in importance.items():
                if feat not in combined_scores:
                    combined_scores[feat] = {'scores': [], 'nan_count': 0}
                
                if score is None or (isinstance(score, float) and np.isnan(score)):
                    combined_scores[feat]['nan_count'] += 1
                    combined_scores[feat]['scores'].append(0.0)  # Penalize NaN with 0
                else:
                    normalized = abs(score) / max_score
                    combined_scores[feat]['scores'].append(normalized)

    # Calculate average importance score with penalty for NaN/missing
    average_scores = {}
    for feature, data in combined_scores.items():
        scores = data['scores']
        nan_count = data['nan_count']
        
        if not scores:
            average_scores[feature] = 0.0
        else:
            # Average score with implicit penalty (NaN counted as 0)
            avg = sum(scores) / len(scores)
            # Additional penalty if feature had NaN in multiple methods
            # This ensures features with consistent scores rank higher
            penalty = 1.0 - (nan_count / method_count) * 0.5  # Up to 50% penalty
            average_scores[feature] = avg * penalty

    # Sort features by average importance score
    sorted_features = sorted(average_scores.items(), key=lambda x: x[1], reverse=True)

    return sorted_features

def get_available_projects(level):
    """Get list of available projects for a given level"""
    if level == 'commit':
        base_dir = 'commit_data'
    elif level == 'file':
        base_dir = 'file_data'
    elif level == 'method':
        base_dir = 'method_data'
    else:
        return []

    if not os.path.exists(base_dir):
        return []

    return [d for d in os.listdir(base_dir)
            if os.path.isdir(os.path.join(base_dir, d)) and d not in ['combined', 'combine']]

def run_feature_selection(project, level, method='all', k_features=None):
    """Run feature selection for a specific project, level and method"""
    print(f"\n{'='*80}")
    print(f"Running feature selection: Project={project}, Level={level}, Method={method}")
    print(f"{'='*80}\n")

    # Determine input and output paths based on level and project
    if level == 'commit':
        input_dir = os.path.join('commit_data', project)
        bug_file = 'bugs.csv'
        non_bug_file = 'non_bugs.csv'
        output_dir = os.path.join('results_commit_level', project)
    elif level == 'file':
        input_dir = os.path.join('file_data', project)
        bug_file = 'file_bug_metrics.csv'
        non_bug_file = 'file_non_bug_metrics.csv'
        output_dir = os.path.join('results_file_level', project)
    elif level == 'method':
        input_dir = os.path.join('method_data', project)
        bug_file = 'method_bug_metrics.csv'
        non_bug_file = 'method_non_bug_metrics.csv'
        output_dir = os.path.join('results_method_level', project)
    else:
        print(f"Error: Invalid level '{level}'")
        return False

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Load the data
    bug_path = os.path.join(input_dir, bug_file)
    non_bug_path = os.path.join(input_dir, non_bug_file)

    if not os.path.exists(bug_path) or not os.path.exists(non_bug_path):
        print(f"Skipping {project} at {level} level - required files not found in {input_dir}")
        print(f"Missing: {bug_path}" if not os.path.exists(bug_path) else "")
        print(f"Missing: {non_bug_path}" if not os.path.exists(non_bug_path) else "")
        return False

    # Read data
    try:
        bug_df = pd.read_csv(bug_path)
        non_bug_df = pd.read_csv(non_bug_path)
    except Exception as e:
        print(f"Error reading data files for {project} at {level} level: {str(e)}")
        return False

    # Add labels
    bug_df['is_bug'] = 1
    non_bug_df['is_bug'] = 0

    # Combine datasets
    df = pd.concat([bug_df, non_bug_df], ignore_index=True)

    # Ensure we have enough data
    if len(df) < 2:
        print(f"Skipping {project} at {level} level - not enough data (only {len(df)} records)")
        return False

    # Prepare feature set
    # Use all numeric columns as features except obvious non-feature columns
    exclude_cols = ['is_bug', 'file', 'path', 'commit', 'method', 'class', 'id',
                   'package', 'repo', 'timestamp', 'date']

    # First identify all potential feature columns (exclude certain columns)
    potential_features = [col for col in df.columns if col not in exclude_cols]

    # Then filter to keep only numeric and boolean columns
    feature_columns = []
    for col in potential_features:
        try:
            # Check if column contains numeric or boolean data
            if pd.api.types.is_numeric_dtype(df[col]):
                # Check for constant columns (zero variance) and exclude them
                if len(df[col].unique()) > 1:
                    feature_columns.append(col)
                else:
                    print(f"  Skipping column '{col}' - constant value")
        except:
            print(f"  Skipping non-numeric column '{col}'")

    if not feature_columns:
        print(f"Skipping {project} at {level} level - no usable feature columns found")
        return False

    print(f"Using {len(feature_columns)} features: {', '.join(feature_columns[:5])}...")

    # Replace NaN values with 0 for all feature columns
    df[feature_columns] = df[feature_columns].fillna(0)

    # Remove outliers using IQR method
    def remove_outliers(df, columns):
        for column in columns:
            # Skip boolean columns
            if df[column].dtype == bool or set(df[column].unique()).issubset({0, 1, True, False}):
                continue

            try:
                Q1 = df[column].quantile(0.25)
                Q3 = df[column].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                df[column] = df[column].clip(lower=lower_bound, upper=upper_bound)
            except (TypeError, ValueError) as e:
                print(f"  Warning: Could not process outliers for column '{column}': {str(e)}")
        return df

    df = remove_outliers(df, feature_columns)

    X = df[feature_columns]
    y = df['is_bug']

    methods_to_run = []
    if method == 'all':
        methods_to_run = ['variance', 'chi2', 'rfe', 'lasso', 'rf', 'mi']
    else:
        methods_to_run = [method]

    all_results = {}

    for m in methods_to_run:
        print(f"\nApplying {m} feature selection...")
        try:
            # Skip if we have too few features for this method
            if len(feature_columns) < 2 and m in ['rfe', 'lasso']:
                print(f"  Skipping {m} - requires at least 2 features")
                continue

            # Handle class imbalance check
            if len(y.unique()) < 2:
                print(f"  Skipping {m} - needs samples from both classes")
                continue

            results = select_features(X, y, method=m, k=k_features)

            # If no features were selected, continue to next method
            if not results[1]:
                print(f"  No features selected by {m}, skipping...")
                continue

            all_results[m] = results

            # Save individual method results
            analyze_feature_selection_results(df, results[1], results[2], m, output_dir)
        except Exception as e:
            print(f"  Error during {m} feature selection: {str(e)}")
            continue

    if all_results:
        if len(methods_to_run) > 1:
            # Combine and analyze results from all methods
            sorted_features = combine_feature_importance(all_results)

            # Save combined results
            combined_results_path = os.path.join(output_dir, 'feature_selection', 'combined_results.txt')
            with open(combined_results_path, 'w') as f:
                f.write(f"Combined Feature Importance Rankings for {project} at {level} level:\n")
                f.write("-" * 50 + "\n")
                for feature, score in sorted_features:
                    f.write(f"{feature}: {score:.4f}\n")

            print("\nCombined Feature Rankings (Top 10):")
            print("-" * 50)
            for feature, score in sorted_features[:min(10, len(sorted_features))]:
                print(f"{feature}: {score:.4f}")

            print(f"\nFull results have been saved to {combined_results_path}")
        else:
            m = methods_to_run[0]
            results = all_results[m]
            print(f"\nSelected {len(results[1])} features:")
            for feature in results[1][:min(10, len(results[1]))]:
                score = results[2].get(feature, 'N/A')
                print(f"- {feature}: {score}")

        print(f"\nResults have been saved to {os.path.join(output_dir, 'feature_selection')}")
        return True

    return False

def main():
    parser = argparse.ArgumentParser(description='Feature selection for bug prediction analysis')
    parser.add_argument('--method', type=str, default='all',
                       choices=['all', 'variance', 'chi2', 'rfe', 'lasso', 'rf', 'mi'],
                       help='Feature selection method to use (default: all)')
    parser.add_argument('--k-features', type=int,
                       help='Number of features to select (default: half of features)')
    parser.add_argument('--project', type=str,
                       help='Project name (e.g., influxdb, caddy, etc.). If not specified, run on all projects.')
    parser.add_argument('--level', type=str,
                       choices=['commit', 'file', 'method'],
                       help='Analysis level (commit, file, or method). If not specified, run on all levels.')
    args = parser.parse_args()

    # Determine levels to process
    levels = ['commit', 'file', 'method'] if args.level is None else [args.level]

    # Track overall success
    total_runs = 0
    successful_runs = 0

    # Process each level
    for level in levels:
        print(f"\n\n{'#'*100}")
        print(f"# Processing {level.upper()} level analysis")
        print(f"{'#'*100}\n")

        # Determine projects to process
        if args.project:
            projects = [args.project]
        else:
            projects = get_available_projects(level)
            if not projects:
                print(f"No projects found for {level} level analysis")
                continue

        print(f"Found {len(projects)} projects for {level} level: {', '.join(projects)}")

        # Process each project
        for project in projects:
            total_runs += 1
            success = run_feature_selection(
                project=project,
                level=level,
                method=args.method,
                k_features=args.k_features
            )
            if success:
                successful_runs += 1

    print(f"\n{'#'*100}")
    print(f"Feature selection complete: {successful_runs}/{total_runs} runs successful")
    print(f"{'#'*100}")

if __name__ == "__main__":
    main()
