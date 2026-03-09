import datetime
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import xgboost as xgb
import lightgbm as lgb
import multiprocessing
import os
import json
from sklearn.model_selection import train_test_split, GridSearchCV, KFold, TimeSeriesSplit
from sklearn.base import BaseEstimator
from sklearn.naive_bayes import GaussianNB
# from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc, roc_auc_score, matthews_corrcoef, average_precision_score
from sklearn.tree import DecisionTreeClassifier # Added
from sklearn.neural_network import MLPClassifier # Added
from sklearn.neighbors import NearestNeighbors
from catboost import CatBoostClassifier
from imblearn.over_sampling import SMOTE, RandomOverSampler, ADASYN, BorderlineSMOTE
from imblearn.under_sampling import RandomUnderSampler, NearMiss, TomekLinks
from imblearn.combine import SMOTETomek, SMOTEENN
from imblearn.pipeline import Pipeline as ImbPipeline  # For resampling in pipeline
from pathlib import Path
import contextlib
import tabulate
from tqdm.rich import tqdm
from sklearn.utils import shuffle
from scipy.stats import norm
from scipy.stats import friedmanchisquare, wilcoxon
import scikit_posthocs as sp # Added for Nemenyi test
import autorank
import feature_select as fs # Added for dynamic feature selection
import warnings
warnings.filterwarnings('ignore')
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn, TimeElapsedColumn, MofNCompleteColumn
from sklearn.base import clone

# Optuna-based hyperparameter tuning (new unified protocol)
from optuna_tuning import (
    OptunaHyperparameterTuner,
    TuningConfig,
    get_adaptive_inner_cv,
    RANDOM_SEED as OPTUNA_RANDOM_SEED,
    N_TRIALS as OPTUNA_N_TRIALS
)

# --- Global Definitions ---
ALL_LEVELS = ['commit', 'file', 'method']

# =============================================================================
# Dataset Quality Thresholds for Reliable Statistical Analysis
# =============================================================================
# These thresholds determine which projects are included in PRIMARY statistical
# analyses (model ranking, Friedman/Nemenyi tests, CD diagrams). Projects below
# these thresholds are reported as EXPLORATORY (included for completeness but
# excluded from main comparative conclusions).
#
# Key insight: The critical factor for reliable metrics in imbalanced datasets
# is not total sample count (N) but MINORITY CLASS COUNT (buggy instances).
# With very few buggy samples, metrics like F1/MCC can fluctuate dramatically
# based on single sample changes.
#
# References for threshold choices:
# - Minimum 20 minority samples: Common heuristic ensuring at least a few
#   minority instances per CV fold and meaningful holdout evaluation
# - These are CONSERVATIVE thresholds; more stringent values (30+) would
#   reduce variance further but exclude more projects

# Import all thresholds from centralized module for consistency
from adequacy_thresholds import (
    MIN_TRAINVAL_MINORITY_PRIMARY,
    MIN_HOLDOUT_MINORITY_PRIMARY,
    MIN_TRAINVAL_MINORITY_EXPLORATORY,
    MIN_HOLDOUT_MINORITY_EXPLORATORY,
    MIN_TRAINVAL_SAMPLES,
    MIN_HOLDOUT_SAMPLES,
    MIN_CV_FOLD_SAMPLES,
    MIN_CV_FOLD_MINORITY
)

# Dataset quality classification
class DatasetQuality:
    """
    Enumeration for dataset quality levels.

    These categories determine inclusion in statistical analyses:
    - PRIMARY: Included in main statistical comparisons (Friedman/Nemenyi, CD diagrams)
    - EXPLORATORY: Reported for completeness but not in primary comparisons
    - INSUFFICIENT: Metrics would be unreliable due to extremely limited minority samples

    IMPORTANT: Thresholds are defined a priori and depend ONLY on class counts,
    not on model performance. This prevents any cherry-picking concerns.
    """
    PRIMARY = "primary"           # Suitable for main statistical analyses
    EXPLORATORY = "exploratory"   # Reported but not in primary comparisons
    INSUFFICIENT = "insufficient" # Not included in primary comparisons (limited minority)


def assess_dataset_quality(n_trainval: int, n_trainval_minority: int,
                           n_holdout: int, n_holdout_minority: int) -> tuple:
    """
    Assess dataset quality based on minority class counts.

    This function implements a priori thresholds that depend ONLY on class counts,
    not on model performance, to avoid any cherry-picking concerns.

    Thresholds:
        PRIMARY: Train+Val buggy >= 20 AND Holdout buggy >= 10
        EXPLORATORY: Train+Val buggy >= 5 AND Holdout buggy >= 3
        INSUFFICIENT: Below exploratory thresholds

    Returns:
        tuple: (quality_level, reasons_dict)

    Quality levels:
        - PRIMARY: Included in main statistical analyses (Friedman/Nemenyi, CD diagrams)
        - EXPLORATORY: Reported for completeness but not in primary comparisons
        - INSUFFICIENT: Not included in primary comparisons due to extremely limited minority
    """
    reasons = {
        'trainval_samples': n_trainval,
        'trainval_minority': n_trainval_minority,
        'holdout_samples': n_holdout,
        'holdout_minority': n_holdout_minority,
        'issues': []
    }

    # Check for INSUFFICIENT (technical guardrails + very low minority counts)
    if n_trainval < MIN_TRAINVAL_SAMPLES:
        reasons['issues'].append(f"Train+Val samples ({n_trainval}) < {MIN_TRAINVAL_SAMPLES}")
        return DatasetQuality.INSUFFICIENT, reasons

    if n_holdout < MIN_HOLDOUT_SAMPLES:
        reasons['issues'].append(f"Holdout samples ({n_holdout}) < {MIN_HOLDOUT_SAMPLES}")
        return DatasetQuality.INSUFFICIENT, reasons

    if n_trainval_minority < MIN_TRAINVAL_MINORITY_EXPLORATORY:
        reasons['issues'].append(f"Train+Val buggy ({n_trainval_minority}) < {MIN_TRAINVAL_MINORITY_EXPLORATORY}")
        return DatasetQuality.INSUFFICIENT, reasons

    if n_holdout_minority < MIN_HOLDOUT_MINORITY_EXPLORATORY:
        reasons['issues'].append(f"Holdout buggy ({n_holdout_minority}) < {MIN_HOLDOUT_MINORITY_EXPLORATORY}")
        return DatasetQuality.INSUFFICIENT, reasons

    # Check for EXPLORATORY
    if n_trainval_minority < MIN_TRAINVAL_MINORITY_PRIMARY:
        reasons['issues'].append(f"Train+Val minority ({n_trainval_minority}) < {MIN_TRAINVAL_MINORITY_PRIMARY} (exploratory)")
        return DatasetQuality.EXPLORATORY, reasons

    if n_holdout_minority < MIN_HOLDOUT_MINORITY_PRIMARY:
        reasons['issues'].append(f"Holdout minority ({n_holdout_minority}) < {MIN_HOLDOUT_MINORITY_PRIMARY} (exploratory)")
        return DatasetQuality.EXPLORATORY, reasons

    # All checks passed - PRIMARY quality
    return DatasetQuality.PRIMARY, reasons


def get_dataset_quality_summary(projects_quality: dict) -> dict:
    """
    Summarize dataset quality across all projects.

    Args:
        projects_quality: Dict mapping (project, level) to (quality, reasons)

    Returns:
        Summary dict with counts and lists per quality level
    """
    summary = {
        DatasetQuality.PRIMARY: [],
        DatasetQuality.EXPLORATORY: [],
        DatasetQuality.INSUFFICIENT: []
    }

    for (project, level), (quality, reasons) in projects_quality.items():
        if quality not in summary:
            summary[quality] = []
        summary[quality].append({
            'project': project,
            'level': level,
            'reasons': reasons
        })

    return {
        'primary_count': len(summary[DatasetQuality.PRIMARY]),
        'exploratory_count': len(summary[DatasetQuality.EXPLORATORY]),
        'insufficient_count': len(summary[DatasetQuality.INSUFFICIENT]),
        'primary_projects': summary[DatasetQuality.PRIMARY],
        'exploratory_projects': summary[DatasetQuality.EXPLORATORY],
        'insufficient_projects': summary[DatasetQuality.INSUFFICIENT]
    }

# Define all available ACTUAL resampling strategies (excluding 'none')
ALL_ACTUAL_RESAMPLING_METHODS = [
    'smote', 'random_under', 'near_miss', 'tomek', 'random_over',
    'adasyn', 'borderline', 'smote_tomek', 'smote_enn', 'rose'
]

# Define all classifier function base names (used for reconstructing results)
ALL_CLASSIFIER_FUNCTION_NAMES = [
    'naive_bayes', 'xgboost', 'random_forest', 'logistic_regression',
    'catboost', 'lightgbm', 'gradient_boosting', 'decision_tree',
    'voting', 'mlp', 'stacking'
]

# --- Configuration ---
BASE_DIR = Path(__file__).resolve().parent
LOG_DIR = BASE_DIR / 'log'
COMMIT_DATA_DIR = BASE_DIR / 'commit_data'
METHOD_DATA_DIR = BASE_DIR / 'method_data'
FILE_DATA_DIR = BASE_DIR / 'file_data'

RESULTS_COMMIT_LEVEL_DIR = BASE_DIR / 'results_commit_level'
RESULTS_METHOD_LEVEL_DIR = BASE_DIR / 'results_method_level'
RESULTS_FILE_LEVEL_DIR = BASE_DIR / 'results_file_level'

# --- Logger Setup ---
LOG_DIR.mkdir(parents=True, exist_ok=True)
timestamp = datetime.datetime.now().strftime("%d%m%Y-%H%M%S")
log_file_name = f"analiz-{timestamp}.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_DIR / log_file_name)
        # Note: StreamHandler removed - logs only go to file now
    ]
)

# --- Resampling Method Getter ---
def get_resampling_method(strategy_name, random_state=42):
    """Returns a resampling method instance based on its name."""
    if strategy_name == 'smote':
        return SMOTE(random_state=random_state)
    elif strategy_name == 'random_under':
        return RandomUnderSampler(random_state=random_state)
    elif strategy_name == 'near_miss':
        return NearMiss() # NearMiss has different versions, default is version 1
    elif strategy_name == 'tomek':
        return TomekLinks()
    elif strategy_name == 'random_over':
        return RandomOverSampler(random_state=random_state)
    elif strategy_name == 'adasyn':
        return ADASYN(random_state=random_state)
    elif strategy_name == 'borderline':
        return BorderlineSMOTE(random_state=random_state)
    elif strategy_name == 'smote_tomek':
        return SMOTETomek(random_state=random_state)
    elif strategy_name == 'smote_enn':
        return SMOTEENN(random_state=random_state)
    elif strategy_name == 'rose':
        return ROSE(random_state=random_state)
    elif strategy_name is None or strategy_name.lower() == 'none':
        return None # No resampling
    else:
        logging.warning(f"Resampling strategy '{strategy_name}' not recognized. Returning None.")
        return None

class ROSE:
    """Random Over Sampling Examples (ROSE)"""
    def __init__(self, random_state=None):
        self.random_state = random_state
        self.feature_names = None

    def fit_resample(self, X, y):
        np.random.seed(self.random_state)
        self.feature_names = X.columns if isinstance(X, pd.DataFrame) else None
        X_np = X.values if isinstance(X, pd.DataFrame) else X
        y_np = y.values if isinstance(y, pd.Series) else y

        X_maj = X_np[y_np == 0]
        X_min = X_np[y_np == 1]
        n_maj, n_min = len(X_maj), len(X_min)

        if n_min == 0:
            logging.warning("ROSE: No minority class samples found. Returning original data.")
            return X, y

        h = np.std(X_min, axis=0) * (4 / (3 * n_min))**(1/5)
        h = np.where(h == 0, 1e-6, h) # Avoid zero bandwidth

        n_samples_to_generate = max(0, n_maj - n_min)
        if n_samples_to_generate == 0: # Already balanced or more minority samples
             return X, y


        indices = np.random.randint(0, n_min, size=n_samples_to_generate)
        noise = np.random.normal(0, 1, (n_samples_to_generate, X_np.shape[1]))
        X_synthetic = X_min[indices] + noise * h

        X_resampled_np = np.vstack([X_np, X_synthetic])
        y_resampled_np = np.hstack([y_np, np.ones(len(X_synthetic))])

        # Shuffle
        indices = np.random.permutation(len(X_resampled_np))
        X_resampled_np = X_resampled_np[indices]
        y_resampled_np = y_resampled_np[indices]


        if self.feature_names is not None:
            X_resampled = pd.DataFrame(X_resampled_np, columns=self.feature_names)
            y_resampled = pd.Series(y_resampled_np, name=y.name if hasattr(y, 'name') else 'is_bug') # Fixed parenthesis
        else:
            X_resampled, y_resampled = X_resampled_np, y_resampled_np

        return X_resampled, y_resampled


# --- Global Class Weight Mode (set by CLI argument) ---
# This is set in main() based on --class-weight argument
_CLASS_WEIGHT_MODE = 'auto'  # 'auto', 'balanced', 'none'


def get_class_weight_value(resampling_strategy=None):
    """
    Determine the class_weight parameter value based on the global mode and resampling strategy.

    Parameters
    ----------
    resampling_strategy : str, optional
        The resampling strategy being used (e.g., 'smote', 'adasyn', 'none', None)

    Returns
    -------
    str or None
        'balanced' to use balanced class weights, None to disable
    """
    global _CLASS_WEIGHT_MODE

    if _CLASS_WEIGHT_MODE == 'balanced':
        # Always use balanced class weights, regardless of resampling
        return 'balanced'
    elif _CLASS_WEIGHT_MODE == 'none':
        # Never use class weights
        return None
    else:  # 'auto' mode (default)
        # Disable class_weight when resampling is used to avoid double intervention
        if resampling_strategy is None or resampling_strategy == 'none':
            return 'balanced'
        else:
            return None


# --- Hierarchical Results Directory Structure ---
# New structure: results_{level}_level/{project}/{cv_type}/{feature_set}/{resampling}/
# cv_type: 'temporal' | 'shuffle'
# feature_set: 'full' | 'no_go_metrics'
# resampling: 'none' | 'smote' | 'adasyn' | etc.

def get_cv_type_name(shuffle_cv=False):
    """Get CV type directory name based on shuffle_cv flag."""
    return 'shuffle' if shuffle_cv else 'temporal'

def get_feature_set_name(exclude_go_metrics=False):
    """Get feature set directory name based on exclude_go_metrics flag."""
    return 'no_go_metrics' if exclude_go_metrics else 'full'

def get_analysis_output_dir(level, project_name, cv_type='temporal', feature_set='full',
                            resampling='none', cli_args=None):
    """
    Get the hierarchical output directory for analysis results.

    New structure:
        results_{level}_level/{project}/{cv_type}/{feature_set}/{resampling}/

    Parameters
    ----------
    level : str
        Granularity level ('commit', 'file', 'method')
    project_name : str
        Name of the project
    cv_type : str
        Cross-validation type: 'temporal' or 'shuffle'
    feature_set : str
        Feature set: 'full' or 'no_go_metrics'
    resampling : str
        Resampling strategy: 'none', 'smote', 'adasyn', etc.
    cli_args : argparse.Namespace, optional
        CLI arguments to extract cv_type and feature_set from

    Returns
    -------
    Path
        Output directory path
    """
    # Override with cli_args if provided
    if cli_args:
        cv_type = get_cv_type_name(getattr(cli_args, 'shuffle_cv', False))
        feature_set = get_feature_set_name(getattr(cli_args, 'exclude_go_metrics', False))

    results_base_dir = get_results_dir(level)
    output_dir = results_base_dir / project_name / cv_type / feature_set / resampling
    return output_dir


def parse_analysis_dir_path(dir_path):
    """
    Parse a hierarchical analysis directory path to extract components.

    Parameters
    ----------
    dir_path : Path
        Path like: results_method_level/influxdb/temporal/full/smote

    Returns
    -------
    dict with keys: level, project, cv_type, feature_set, resampling
    """
    parts = Path(dir_path).parts

    # Find results_*_level in path
    for i, part in enumerate(parts):
        if part.startswith('results_') and part.endswith('_level'):
            level = part.replace('results_', '').replace('_level', '')
            if len(parts) > i + 4:
                return {
                    'level': level,
                    'project': parts[i + 1],
                    'cv_type': parts[i + 2],
                    'feature_set': parts[i + 3],
                    'resampling': parts[i + 4]
                }
    return None


def list_analysis_dirs(level, project_name=None, cv_type=None, feature_set=None, resampling=None):
    """
    List analysis directories matching the given criteria.

    Parameters
    ----------
    level : str
        Granularity level
    project_name : str, optional
        Filter by project name
    cv_type : str, optional
        Filter by CV type ('temporal' or 'shuffle')
    feature_set : str, optional
        Filter by feature set ('full' or 'no_go_metrics')
    resampling : str, optional
        Filter by resampling strategy

    Returns
    -------
    list of Path objects
    """
    results_dir = get_results_dir(level)
    matching_dirs = []

    # Build glob pattern
    project_pattern = project_name if project_name else '*'
    cv_pattern = cv_type if cv_type else '*'
    feature_pattern = feature_set if feature_set else '*'
    resampling_pattern = resampling if resampling else '*'

    pattern = f"{project_pattern}/{cv_pattern}/{feature_pattern}/{resampling_pattern}"

    for path in results_dir.glob(pattern):
        if path.is_dir() and not path.name.startswith('_'):
            matching_dirs.append(path)

    return matching_dirs


class CommitGroupTimeSeriesSplit:
    """
    Time-series cross-validator that respects commit group boundaries.

    This splitter ensures:
    1. All instances from the same commit (sha) stay together in train or test
    2. Training data always comes chronologically before test data
    3. No data leakage between train and test sets

    Parameters
    ----------
    n_splits : int, default=5
        Number of splits/folds
    min_train_ratio : float, default=0.15
        Minimum ratio of data that must be in the training set
    gap : int, default=0
        Number of commit groups to exclude between train and test (gap period)
    min_class_ratio : float, default=0.05
        Minimum ratio of minority class required in both train and test sets.
        Folds that don't meet this threshold are skipped.
    """

    def __init__(self, n_splits=5, min_train_ratio=0.15, gap=0, min_class_ratio=0.05):
        self.n_splits = n_splits
        self.min_train_ratio = min_train_ratio
        self.gap = gap
        self.min_class_ratio = min_class_ratio

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits

    def split(self, X, y=None, groups=None, timestamps=None):
        """
        Generate indices to split data into training and test sets.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data
        y : array-like of shape (n_samples,), optional
            Target variable
        groups : array-like of shape (n_samples,)
            Group labels (commit sha) for the samples
        timestamps : array-like of shape (n_samples,), optional
            Timestamps for ordering. If None, assumes X is already sorted.

        Yields
        ------
        train : ndarray
            Training set indices for this split
        test : ndarray
            Test set indices for this split
        """
        if groups is None:
            raise ValueError("groups (commit sha) must be provided for CommitGroupTimeSeriesSplit")

        n_samples = len(X) if hasattr(X, '__len__') else X.shape[0]
        groups_array = np.array(groups) if not isinstance(groups, np.ndarray) else groups

        # Validate timestamps if provided (data should be sorted)
        if timestamps is not None:
            timestamps_array = np.array(timestamps)
            # Check that timestamps are non-decreasing (allowing ties)
            if not np.all(np.diff(timestamps_array) >= 0):
                logging.warning(
                    "CommitGroupTimeSeriesSplit: timestamps are not sorted! "
                    "Results may violate temporal integrity. Please sort data by timestamp before splitting."
                )

        # Get unique commits in order (assumes data is already sorted by timestamp)
        # Create a mapping of group -> first occurrence index for ordering
        unique_groups_ordered = []
        seen_groups = set()

        for i, g in enumerate(groups_array):
            if g not in seen_groups:
                unique_groups_ordered.append(g)
                seen_groups.add(g)

        n_groups = len(unique_groups_ordered)

        if n_groups < self.n_splits + 1:
            raise ValueError(
                f"Cannot create {self.n_splits} splits with only {n_groups} unique commit groups. "
                f"Need at least {self.n_splits + 1} groups."
            )

        # Convert y to array for class ratio checking
        y_array = np.array(y) if y is not None else None

        # Calculate minimum number of groups for training
        min_train_groups = max(1, int(n_groups * self.min_train_ratio))

        # Calculate test size for each fold (expanding window approach)
        # Each fold gets progressively more training data
        test_groups_per_fold = max(1, (n_groups - min_train_groups) // self.n_splits)

        skipped_folds = []

        for fold in range(self.n_splits):
            # Calculate train end and test start for this fold
            # Expanding window: train grows with each fold
            train_end_group_idx = min_train_groups + fold * test_groups_per_fold
            test_start_group_idx = train_end_group_idx + self.gap
            test_end_group_idx = min(train_end_group_idx + test_groups_per_fold + self.gap, n_groups)

            if test_start_group_idx >= n_groups:
                logging.warning(f"Fold {fold+1}: Not enough groups for test set after gap. Skipping fold.")
                skipped_folds.append((fold+1, "not enough groups"))
                continue

            # Get groups for train and test
            train_groups = set(unique_groups_ordered[:train_end_group_idx])
            test_groups = set(unique_groups_ordered[test_start_group_idx:test_end_group_idx])

            # Get sample indices
            train_indices = np.where(np.isin(groups_array, list(train_groups)))[0]
            test_indices = np.where(np.isin(groups_array, list(test_groups)))[0]

            if len(train_indices) == 0 or len(test_indices) == 0:
                logging.warning(f"Fold {fold+1}: Empty train or test set. Skipping fold.")
                skipped_folds.append((fold+1, "empty set"))
                continue

            # Check minimum class ratio if y is provided
            if y_array is not None and self.min_class_ratio > 0:
                train_y = y_array[train_indices]
                test_y = y_array[test_indices]

                # Calculate class ratios
                train_pos_ratio = np.mean(train_y)
                train_neg_ratio = 1 - train_pos_ratio
                test_pos_ratio = np.mean(test_y)
                test_neg_ratio = 1 - test_pos_ratio

                # Check if minority class meets threshold
                train_min_ratio = min(train_pos_ratio, train_neg_ratio)
                test_min_ratio = min(test_pos_ratio, test_neg_ratio)

                if train_min_ratio < self.min_class_ratio:
                    logging.warning(
                        f"Fold {fold+1}: Train set minority class ratio ({train_min_ratio:.2%}) "
                        f"below threshold ({self.min_class_ratio:.2%}). Skipping fold."
                    )
                    skipped_folds.append((fold+1, f"train minority ratio {train_min_ratio:.2%}"))
                    continue

                if test_min_ratio < self.min_class_ratio:
                    logging.warning(
                        f"Fold {fold+1}: Test set minority class ratio ({test_min_ratio:.2%}) "
                        f"below threshold ({self.min_class_ratio:.2%}). Skipping fold."
                    )
                    skipped_folds.append((fold+1, f"test minority ratio {test_min_ratio:.2%}"))
                    continue

            yield train_indices, test_indices

        if skipped_folds:
            logging.info(f"Skipped folds due to class imbalance: {skipped_folds}")

    def validate_temporal_integrity(self, timestamps, train_idx, test_idx, fold_num=None):
        """
        Validate that train data comes before test data temporally.

        Note: For commits with the same timestamp, we allow train_max == test_min
        since in practice this is rare and the group-based splitting ensures
        commit integrity. The <= check is used for timestamps that might tie.

        Returns
        -------
        dict with validation results
        """
        train_timestamps = np.array(timestamps)[train_idx]
        test_timestamps = np.array(timestamps)[test_idx]

        train_max = np.max(train_timestamps)
        test_min = np.min(test_timestamps)

        # Allow equality for same-second commits (handled by group integrity)
        # Strict violation is only when train has LATER timestamps than test
        is_valid = train_max <= test_min
        has_tie = (train_max == test_min)

        result = {
            'is_valid': is_valid,
            'train_max_timestamp': int(train_max),
            'test_min_timestamp': int(test_min),
            'gap_seconds': int(test_min - train_max),
            'train_size': len(train_idx),
            'test_size': len(test_idx),
            'has_timestamp_tie': has_tie
        }

        if fold_num is not None:
            result['fold'] = fold_num

        if has_tie:
            logging.warning(
                f"Temporal tie detected{'(Fold '+str(fold_num)+')' if fold_num else ''}: "
                f"train_max_timestamp ({train_max}) == test_min_timestamp ({test_min}). "
                f"This is acceptable if group integrity is maintained."
            )

        if not is_valid:
            logging.error(
                f"Temporal integrity violation{'(Fold '+str(fold_num)+')' if fold_num else ''}: "
                f"train_max_timestamp ({train_max}) > test_min_timestamp ({test_min})"
            )

        return result


def get_adaptive_fold_counts(n_samples, n_unique_commits, n_positive_samples):
    """
    Determine adaptive fold counts based on dataset characteristics.

    Parameters
    ----------
    n_samples : int
        Total number of samples
    n_unique_commits : int
        Number of unique commits (groups)
    n_positive_samples : int
        Number of positive (bug) samples

    Returns
    -------
    tuple (outer_folds, inner_folds)
    """
    # Primary criterion: unique commits
    if n_unique_commits < 50 or n_positive_samples < 30:
        outer_folds, inner_folds = 3, 2
    elif n_unique_commits < 150:
        outer_folds, inner_folds = 4, 3
    else:
        outer_folds, inner_folds = 5, 5

    # Ensure we have enough commits for the folds
    # Need at least outer_folds + 1 commits for outer CV
    max_outer_folds = max(2, n_unique_commits - 1)
    outer_folds = min(outer_folds, max_outer_folds)

    # For inner CV, we need fewer commits (working on train subset)
    # Estimate: ~80% of commits available for inner CV
    estimated_inner_commits = int(n_unique_commits * 0.8 * (outer_folds - 1) / outer_folds)
    max_inner_folds = max(2, estimated_inner_commits - 1)
    inner_folds = min(inner_folds, max_inner_folds)

    logging.info(
        f"Adaptive fold selection: {outer_folds} outer folds, {inner_folds} inner folds "
        f"(based on {n_unique_commits} unique commits, {n_positive_samples} positive samples)"
    )

    return outer_folds, inner_folds


def validate_commit_group_integrity(groups, train_idx, test_idx, fold_num=None):
    """
    Validate that no commit appears in both train and test sets.

    Returns
    -------
    dict with validation results
    """
    groups_array = np.array(groups)
    train_groups = set(groups_array[train_idx])
    test_groups = set(groups_array[test_idx])

    overlap = train_groups.intersection(test_groups)
    is_valid = len(overlap) == 0

    result = {
        'is_valid': is_valid,
        'train_unique_commits': len(train_groups),
        'test_unique_commits': len(test_groups),
        'overlapping_commits': list(overlap) if not is_valid else []
    }

    if fold_num is not None:
        result['fold'] = fold_num

    if not is_valid:
        logging.error(
            f"Commit group integrity violation{'(Fold '+str(fold_num)+')' if fold_num else ''}: "
            f"{len(overlap)} commits appear in both train and test: {list(overlap)[:5]}..."
        )

    return result


def get_fold_class_distribution(y, train_idx, test_idx, fold_num=None):
    """
    Get class distribution for train and test sets.

    Returns
    -------
    dict with distribution information
    """
    y_array = np.array(y) if not isinstance(y, np.ndarray) else y

    train_y = y_array[train_idx]
    test_y = y_array[test_idx]

    train_pos_ratio = float(np.sum(train_y) / len(train_y)) if len(train_y) > 0 else 0
    test_pos_ratio = float(np.sum(test_y) / len(test_y)) if len(test_y) > 0 else 0

    result = {
        'train_total': len(train_idx),
        'train_positive': int(np.sum(train_y)),
        'train_negative': int(len(train_y) - np.sum(train_y)),
        'train_positive_ratio': train_pos_ratio,
        'test_total': len(test_idx),
        'test_positive': int(np.sum(test_y)),
        'test_negative': int(len(test_y) - np.sum(test_y)),
        'test_positive_ratio': test_pos_ratio
    }

    if fold_num is not None:
        result['fold'] = fold_num

        # Warn if fold has extreme class imbalance
        if result['train_negative'] == 0:
            logging.warning(
                f"⚠️ [Fold {fold_num}] Training set has NO non-bug samples! "
                f"({result['train_positive']} bugs, 0 non-bugs). "
                f"Model will only learn to predict bugs."
            )
        elif result['train_positive'] == 0:
            logging.warning(
                f"⚠️ [Fold {fold_num}] Training set has NO bug samples! "
                f"(0 bugs, {result['train_negative']} non-bugs). "
                f"Model will only learn to predict non-bugs."
            )
        elif train_pos_ratio > 0.95:
            logging.warning(
                f"⚠️ [Fold {fold_num}] Training set is extremely imbalanced: "
                f"{train_pos_ratio*100:.1f}% bugs ({result['train_positive']} bugs, {result['train_negative']} non-bugs)"
            )
        elif train_pos_ratio < 0.05:
            logging.warning(
                f"⚠️ [Fold {fold_num}] Training set is extremely imbalanced: "
                f"{train_pos_ratio*100:.1f}% bugs ({result['train_positive']} bugs, {result['train_negative']} non-bugs)"
            )

    return result


def ensure_dataframe_after_resampling(X_resampled, y_resampled, original_columns, original_y_name='is_bug'):
    """
    Ensure that resampled data remains as DataFrame/Series with proper column names.

    Some imblearn samplers return numpy arrays, which can cause issues with
    feature importance analysis and scaling operations that expect DataFrame.

    Parameters
    ----------
    X_resampled : array-like
        Resampled feature matrix (may be ndarray or DataFrame)
    y_resampled : array-like
        Resampled target (may be ndarray or Series)
    original_columns : list
        Original column names from the DataFrame
    original_y_name : str
        Original name for the target series

    Returns
    -------
    tuple (X_df, y_series) : properly formatted DataFrame and Series
    """
    # Convert X to DataFrame if needed
    if isinstance(X_resampled, np.ndarray):
        X_df = pd.DataFrame(X_resampled, columns=original_columns)
    else:
        X_df = X_resampled

    # Convert y to Series if needed
    if isinstance(y_resampled, np.ndarray):
        y_series = pd.Series(y_resampled, name=original_y_name)
    else:
        y_series = y_resampled

    return X_df, y_series


def apply_feature_selection_on_fold(X_train, y_train, X_test, fs_method, k_features, output_dir=None, fold_num=None):
    """
    Apply feature selection on fold training data only and transform both train and test.

    This prevents data leakage by ensuring feature selection is performed only on
    the training portion of each fold.

    Parameters
    ----------
    X_train : DataFrame
        Training features for this fold
    y_train : Series
        Training labels for this fold
    X_test : DataFrame
        Test features for this fold
    fs_method : str
        Feature selection method name
    k_features : int
        Number of features to select
    output_dir : Path, optional
        Directory to save feature selection info
    fold_num : int, optional
        Fold number for logging

    Returns
    -------
    tuple (X_train_selected, X_test_selected, selected_features)
    """
    fold_prefix = f"[Fold {fold_num}] " if fold_num else ""

    if X_train.empty or X_train.shape[1] == 0:
        logging.warning(f"{fold_prefix}No features available for feature selection. Skipping.")
        return X_train, X_test, X_train.columns.tolist(), {}

    if len(y_train.unique()) < 2:
        logging.warning(f"{fold_prefix}Only one class in training data. Skipping feature selection.")
        return X_train, X_test, X_train.columns.tolist(), {}

    # Smart k determination if not specified
    # Reference: Guyon & Elisseeff (2003) - "An Introduction to Variable and Feature Selection"
    # recommends using sqrt(n) to 2*sqrt(n) features for high-dimensional data.
    # For smaller feature sets, we use more conservative approaches.
    n_features = X_train.shape[1]
    if k_features:
        k = min(k_features, n_features)
        k_strategy = 'user_specified'
    else:
        # Adaptive k selection based on feature count
        # Small sets (≤10): Keep most features to avoid underfitting
        # Medium sets (11-30): Use ~70% to balance bias-variance
        # Large sets (>30): Use 2*sqrt(n) following Guyon & Elisseeff recommendation
        if n_features <= 10:
            k = max(1, n_features - 1)  # Keep most features for small sets
            k_strategy = 'adaptive_small_n_minus_1'
        elif n_features <= 30:
            k = max(5, int(n_features * 0.7))
            k_strategy = 'adaptive_medium_70pct'
        else:
            k = max(10, int(np.sqrt(n_features) * 2))
            k_strategy = 'adaptive_large_2sqrt_n'

    fs_metadata = {
        'method': fs_method,
        'k_requested': k_features,
        'k_actual': k,
        'k_strategy': k_strategy,
        'k_strategy_reference': 'Guyon & Elisseeff (2003)' if k_strategy.startswith('adaptive') else None,
        'n_features_original': n_features,
        'selected_features': [],
        'feature_scores': {}
    }

    try:
        if fs_method == 'combine':
            all_methods = ['variance', 'chi2', 'rfe', 'lasso', 'rf', 'mi']
            all_results = {}
            for method_name in all_methods:
                try:
                    _, temp_selected, temp_importance = fs.select_features(
                        X_train.copy(), y_train.copy(), method=method_name, k=X_train.shape[1]
                    )
                    if temp_selected:
                        all_results[method_name] = (None, temp_selected, temp_importance)
                except Exception as e:
                    logging.debug(f"{fold_prefix}Error in {method_name} for combine: {e}")

            if all_results:
                combined_ranked = fs.combine_feature_importance(all_results)
                selected_features = [feat for feat, score in combined_ranked[:k]]
                # Store feature scores for metadata
                fs_metadata['feature_scores'] = {feat: float(score) for feat, score in combined_ranked}
            else:
                logging.warning(f"{fold_prefix}Combine strategy failed. Using all features.")
                selected_features = X_train.columns.tolist()
        else:
            _, selected_features, importance_scores = fs.select_features(
                X_train.copy(), y_train.copy(), method=fs_method, k=k
            )
            if importance_scores:
                fs_metadata['feature_scores'] = {k: float(v) if isinstance(v, (int, float, np.floating)) else v
                                                  for k, v in importance_scores.items()}

        if not selected_features:
            logging.warning(f"{fold_prefix}Feature selection returned no features. Using all.")
            selected_features = X_train.columns.tolist()

        # Validate selected features exist in both train and test
        valid_features = [f for f in selected_features if f in X_train.columns and f in X_test.columns]

        if not valid_features:
            logging.warning(f"{fold_prefix}No valid features after selection. Using all.")
            valid_features = X_train.columns.tolist()

        X_train_selected = X_train[valid_features]
        X_test_selected = X_test[valid_features]

        # Update metadata
        fs_metadata['selected_features'] = valid_features
        fs_metadata['n_features_selected'] = len(valid_features)

        logging.info(f"{fold_prefix}Feature selection ({fs_method}): {len(valid_features)} features selected from {n_features} (k={k}, strategy={k_strategy})")

        return X_train_selected, X_test_selected, valid_features, fs_metadata

    except Exception as e:
        logging.error(f"{fold_prefix}Feature selection error: {e}. Using all features.")
        fs_metadata['error'] = str(e)
        return X_train, X_test, X_train.columns.tolist(), fs_metadata


class InnerTemporalCV:
    """
    Time-aware cross-validator for inner CV (hyperparameter tuning).

    Uses a simple time-series split approach for the inner loop, ensuring
    that even within the outer fold's training data, temporal order is respected.
    """

    def __init__(self, n_splits=3, groups=None, timestamps=None):
        self.n_splits = n_splits
        self.groups = groups
        self.timestamps = timestamps

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits

    def split(self, X, y=None, groups=None):
        """
        Generate time-aware splits for inner CV.

        If groups and timestamps were provided at init, uses group-aware splitting.
        Otherwise, falls back to simple sequential splitting.
        """
        n_samples = len(X) if hasattr(X, '__len__') else X.shape[0]

        if self.groups is not None and len(self.groups) == n_samples:
            # Group-aware splitting
            groups_array = np.array(self.groups)
            unique_groups = []
            seen = set()
            for g in groups_array:
                if g not in seen:
                    unique_groups.append(g)
                    seen.add(g)

            n_groups = len(unique_groups)
            if n_groups < self.n_splits + 1:
                # Fall back to fewer splits
                actual_splits = max(2, n_groups - 1)
            else:
                actual_splits = self.n_splits

            # Divide groups into train/val portions for each split
            groups_per_fold = max(1, n_groups // (actual_splits + 1))

            for fold in range(actual_splits):
                train_end_idx = groups_per_fold * (fold + 1)
                val_end_idx = min(train_end_idx + groups_per_fold, n_groups)

                train_groups = set(unique_groups[:train_end_idx])
                val_groups = set(unique_groups[train_end_idx:val_end_idx])

                train_indices = np.where(np.isin(groups_array, list(train_groups)))[0]
                val_indices = np.where(np.isin(groups_array, list(val_groups)))[0]

                if len(train_indices) > 0 and len(val_indices) > 0:
                    yield train_indices, val_indices
        else:
            # Simple sequential splitting (TimeSeriesSplit-like)
            fold_size = n_samples // (self.n_splits + 1)
            for fold in range(self.n_splits):
                train_end = fold_size * (fold + 1)
                val_end = min(train_end + fold_size, n_samples)

                train_indices = np.arange(train_end)
                val_indices = np.arange(train_end, val_end)

                if len(train_indices) > 0 and len(val_indices) > 0:
                    yield train_indices, val_indices


def get_data_dir(level):
    if level == 'commit':
        return COMMIT_DATA_DIR
    elif level == 'method':
        return METHOD_DATA_DIR
    elif level == 'file':
        return FILE_DATA_DIR
    else:
        raise ValueError(f"Invalid level: {level}")

def get_results_dir(level, cpdp_mode=False, lopo_mode=False, source_project=None, destination_project=None):
    base_dir = Path(__file__).resolve().parent / f"results_{level}_level"

    if lopo_mode:
        return base_dir / "_lopo"

    if cpdp_mode:
        if not source_project or not destination_project:
            raise ValueError("Source and destination projects must be provided for CPDP results directory.")
        # Format: results_level/_cpdp/source_to_dest/
        return base_dir / "_cpdp" / f"{source_project}_to_{destination_project}"

    if level == 'commit':
        return RESULTS_COMMIT_LEVEL_DIR
    elif level == 'method':
        return RESULTS_METHOD_LEVEL_DIR
    elif level == 'file':
        return RESULTS_FILE_LEVEL_DIR
    else:
        raise ValueError(f"Invalid level: {level}")

def load_project_data(project_name, level, sort_by_time=True, overlap_only=False):
    """
    Load and combine data for a specific project and level.

    Parameters
    ----------
    project_name : str
        Name of the project to load
    level : str
        Granularity level ('commit', 'file', or 'method')
    sort_by_time : bool, default=True
        If True, sort data chronologically by commit_timestamp and sha
    overlap_only : bool, default=False
        If True, only use data from the overlapping time period where both
        bugs and non-bugs exist. This is useful when non-bug data collection
        started later than bug data collection.

    Returns
    -------
    pd.DataFrame or None
        Combined DataFrame with bugs and non-bugs, sorted by time if requested.
        Includes 'sha' and 'commit_timestamp' columns for temporal splitting.
    """
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
        logging.error(f"Invalid level specified: {level}")
        return None

    bugs_path = project_data_dir / bugs_file
    non_bugs_path = project_data_dir / non_bugs_file

    if not (bugs_path.exists() and non_bugs_path.exists()):
        logging.warning(f"Data files not found for project {project_name} at level {level} in {project_data_dir}")
        return None

    bugs_df = pd.read_csv(bugs_path)
    non_bugs_df = pd.read_csv(non_bugs_path)

    # Check if data files are empty (only headers)
    if len(bugs_df) == 0 and len(non_bugs_df) == 0:
        logging.warning(f"Both data files for project {project_name} at level {level} are empty (only headers). No data to analyze.")
        return None
    elif len(bugs_df) == 0:
        logging.warning(f"Bug data file for project {project_name} at level {level} is empty. Only non-bug data available.")
    elif len(non_bugs_df) == 0:
        logging.warning(f"Non-bug data file for project {project_name} at level {level} is empty. Only bug data available.")

    # === TEMPORAL OVERLAP CHECK ===
    # Check if bugs and non-bugs have overlapping time periods
    if 'commit_timestamp' in bugs_df.columns and 'commit_timestamp' in non_bugs_df.columns:
        bug_min_ts = bugs_df['commit_timestamp'].min()
        bug_max_ts = bugs_df['commit_timestamp'].max()
        nonbug_min_ts = non_bugs_df['commit_timestamp'].min()
        nonbug_max_ts = non_bugs_df['commit_timestamp'].max()

        # Calculate overlap period
        overlap_start = max(bug_min_ts, nonbug_min_ts)
        overlap_end = min(bug_max_ts, nonbug_max_ts)

        from datetime import datetime
        bug_min_date = datetime.fromtimestamp(bug_min_ts).strftime('%Y-%m-%d')
        bug_max_date = datetime.fromtimestamp(bug_max_ts).strftime('%Y-%m-%d')
        nonbug_min_date = datetime.fromtimestamp(nonbug_min_ts).strftime('%Y-%m-%d')
        nonbug_max_date = datetime.fromtimestamp(nonbug_max_ts).strftime('%Y-%m-%d')

        logging.info(f"Temporal coverage - Bugs: {bug_min_date} to {bug_max_date}, Non-bugs: {nonbug_min_date} to {nonbug_max_date}")

        if overlap_start > overlap_end:
            logging.error(
                f"⚠️ CRITICAL: No temporal overlap between bugs and non-bugs for {project_name}! "
                f"Bugs: {bug_min_date} to {bug_max_date}, Non-bugs: {nonbug_min_date} to {nonbug_max_date}. "
                f"Temporal CV will not work correctly!"
            )
        else:
            overlap_start_date = datetime.fromtimestamp(overlap_start).strftime('%Y-%m-%d')
            overlap_end_date = datetime.fromtimestamp(overlap_end).strftime('%Y-%m-%d')

            # Count samples in overlap period
            bugs_in_overlap = len(bugs_df[(bugs_df['commit_timestamp'] >= overlap_start) &
                                          (bugs_df['commit_timestamp'] <= overlap_end)])
            nonbugs_in_overlap = len(non_bugs_df[(non_bugs_df['commit_timestamp'] >= overlap_start) &
                                                  (non_bugs_df['commit_timestamp'] <= overlap_end)])

            # Calculate gap in years
            gap_years = (nonbug_min_ts - bug_min_ts) / (365.25 * 24 * 3600)

            if gap_years > 1:
                logging.warning(
                    f"⚠️ TEMPORAL GAP DETECTED: Non-bug data starts {gap_years:.1f} years after bug data. "
                    f"Overlapping period: {overlap_start_date} to {overlap_end_date} "
                    f"({bugs_in_overlap} bugs, {nonbugs_in_overlap} non-bugs in overlap). "
                    f"Early folds may have imbalanced class distribution!"
                )
                if not overlap_only:
                    logging.info(
                        "💡 TIP: Use --overlap-only flag to use only the overlapping time period "
                        "for more balanced temporal CV folds."
                    )
            else:
                logging.info(
                    f"Overlapping period: {overlap_start_date} to {overlap_end_date} "
                    f"({bugs_in_overlap} bugs, {nonbugs_in_overlap} non-bugs)"
                )

            # Filter to overlap period if requested
            if overlap_only and overlap_start <= overlap_end:
                original_bugs = len(bugs_df)
                original_nonbugs = len(non_bugs_df)

                bugs_df = bugs_df[(bugs_df['commit_timestamp'] >= overlap_start) &
                                  (bugs_df['commit_timestamp'] <= overlap_end)]
                non_bugs_df = non_bugs_df[(non_bugs_df['commit_timestamp'] >= overlap_start) &
                                          (non_bugs_df['commit_timestamp'] <= overlap_end)]

                logging.info(
                    f"🔄 OVERLAP-ONLY MODE: Filtered to overlapping period ({overlap_start_date} to {overlap_end_date}). "
                    f"Bugs: {original_bugs} → {len(bugs_df)}, Non-bugs: {original_nonbugs} → {len(non_bugs_df)}"
                )

    bugs_df['is_bug'] = 1
    non_bugs_df['is_bug'] = 0

    combined_df = pd.concat([bugs_df, non_bugs_df], ignore_index=True)

    # Sort by timestamp (primary) and sha (secondary) for temporal ordering
    if sort_by_time:
        if 'commit_timestamp' not in combined_df.columns:
            logging.warning(f"'commit_timestamp' column not found for {project_name} at {level}. Cannot sort temporally.")
        elif 'sha' not in combined_df.columns:
            logging.warning(f"'sha' column not found for {project_name} at {level}. Sorting by timestamp only.")
            combined_df = combined_df.sort_values(by='commit_timestamp', ascending=True).reset_index(drop=True)
            logging.info(f"Data sorted chronologically by commit_timestamp.")
        else:
            combined_df = combined_df.sort_values(
                by=['commit_timestamp', 'sha'],
                ascending=[True, True]
            ).reset_index(drop=True)
            logging.info(f"Data sorted chronologically by commit_timestamp and sha.")

            # Log temporal range info
            min_ts = combined_df['commit_timestamp'].min()
            max_ts = combined_df['commit_timestamp'].max()
            n_unique_commits = combined_df['sha'].nunique()
            logging.info(
                f"Temporal range: {min_ts} to {max_ts} "
                f"({n_unique_commits} unique commits)"
            )

    logging.info(f"Loaded data for project {project_name} at level {level}. Total samples: {len(combined_df)} (Bugs: {len(bugs_df)}, Non-bugs: {len(non_bugs_df)})")
    return combined_df

def prepare_features(df, level, selected_features_config=None, cli_args=None):
    """Prepare feature set based on the level."""
    if level == 'commit':
        feature_columns = [
            'modified_files_count', 'code_churn', 'max_file_churn', 'avg_file_churn',
            'deletions', 'insertions', 'net_lines', 'dmm_unit_size',
            'dmm_unit_complexity', 'dmm_unit_interfacing', 'total_token_count',
            'total_nloc', 'total_complexity', 'total_changed_method_count'
        ]
    elif level == 'file':
        default_feature_columns = [
            'nloc', 'complexity', 'token_count', 'method_count', 'commit_count',
            'authors_count', 'avg_method_param_count', 'import_count', 'cyclo_per_loc',
            'comment_ratio', 'struct_count', 'interface_count', 'loop_count',
            'error_handling_count', 'goroutine_count', 'channel_count', 'defer_count',
            'context_usage_count', 'json_tag_count', 'variadic_function_count',
            'pointer_receiver_count', 'avg_method_complexity', 'avg_methods_token_count'
        ]
        feature_columns = selected_features_config if selected_features_config else default_feature_columns
    elif level == 'method':
        default_feature_columns = [
            'cyclomatic_complexity', 'nloc', 'token_count', 'parameter_count',
            'defer_count', 'channel_count', 'goroutine_count',
            'error_handling_count', 'loop_count'
        ]
        feature_columns = selected_features_config if selected_features_config else default_feature_columns
    else:
        logging.error(f"Invalid level for feature preparation: {level}")
        return None, None

    # Exclude Go-specific metrics if requested
    if cli_args and cli_args.exclude_go_metrics:
        go_specific_metrics = [
            # File-level & Method-level Go-specific metrics
            'struct_count', 'interface_count', 'goroutine_count', 'channel_count',
            'defer_count', 'context_usage_count', 'json_tag_count',
            'variadic_function_count', 'pointer_receiver_count',
            'error_handling_count'  # Considered Go-specific due to implementation
        ]

        original_feature_count = len(feature_columns)
        feature_columns = [col for col in feature_columns if col not in go_specific_metrics]
        excluded_count = original_feature_count - len(feature_columns)
        if excluded_count > 0:
            logging.info(f"Excluded {excluded_count} Go-specific metrics for level '{level}' due to --exclude-go-metrics flag.")

    # Ensure all feature columns exist, fill missing ones with 0 or log a warning
    for col in feature_columns:
        if col not in df.columns:
            logging.warning(f"Feature column '{col}' not found in DataFrame for level '{level}'. It will be filled with 0.")
            df[col] = 0

    df_filled = df.fillna({col: 0 for col in feature_columns})
    X = df_filled[feature_columns]
    y = df_filled['is_bug']
    return X, y


# =============================================================================
# Bootstrap Confidence Interval Functions
# =============================================================================

def bootstrap_ci(y_true, y_pred, y_prob, metric_func, n_bootstrap=1000, ci_level=0.95, random_state=42):
    """
    Calculate bootstrap confidence interval for a metric.

    Parameters
    ----------
    y_true : array-like
        True labels
    y_pred : array-like
        Predicted labels (for MCC, F1, etc.)
    y_prob : array-like
        Predicted probabilities (for PR-AUC, ROC-AUC)
    metric_func : callable
        Function that takes (y_true, y_pred) or (y_true, y_prob) and returns metric value
    n_bootstrap : int
        Number of bootstrap iterations
    ci_level : float
        Confidence level (e.g., 0.95 for 95% CI)
    random_state : int
        Random seed for reproducibility

    Returns
    -------
    dict with keys: point_estimate, ci_lower, ci_upper, std, n_bootstrap
    """
    np.random.seed(random_state)
    n_samples = len(y_true)

    if n_samples < 10:
        # Too few samples for meaningful bootstrap
        return {
            'point_estimate': np.nan,
            'ci_lower': np.nan,
            'ci_upper': np.nan,
            'std': np.nan,
            'n_bootstrap': 0,
            'warning': 'Too few samples for bootstrap CI'
        }

    bootstrap_scores = []

    for _ in range(n_bootstrap):
        # Sample with replacement
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        y_true_boot = np.array(y_true)[indices]
        y_pred_boot = np.array(y_pred)[indices] if y_pred is not None else None
        y_prob_boot = np.array(y_prob)[indices] if y_prob is not None else None

        # Skip if only one class in bootstrap sample
        if len(np.unique(y_true_boot)) < 2:
            continue

        try:
            # Determine which input to use based on metric function signature
            if y_prob_boot is not None and 'prob' in metric_func.__name__.lower() or 'auc' in metric_func.__name__.lower():
                score = metric_func(y_true_boot, y_prob_boot)
            else:
                score = metric_func(y_true_boot, y_pred_boot)

            if not np.isnan(score) and not np.isinf(score):
                bootstrap_scores.append(score)
        except Exception:
            continue

    if len(bootstrap_scores) < n_bootstrap * 0.5:
        # Too many failed bootstrap iterations
        return {
            'point_estimate': np.nan,
            'ci_lower': np.nan,
            'ci_upper': np.nan,
            'std': np.nan,
            'n_bootstrap': len(bootstrap_scores),
            'warning': f'Only {len(bootstrap_scores)}/{n_bootstrap} bootstrap iterations succeeded'
        }

    # Calculate percentiles
    alpha = 1 - ci_level
    ci_lower = np.percentile(bootstrap_scores, alpha / 2 * 100)
    ci_upper = np.percentile(bootstrap_scores, (1 - alpha / 2) * 100)

    return {
        'point_estimate': np.mean(bootstrap_scores),
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'std': np.std(bootstrap_scores),
        'n_bootstrap': len(bootstrap_scores)
    }


def compute_all_bootstrap_cis(y_true, y_pred, y_prob, n_bootstrap=1000, ci_level=0.95, random_state=42):
    """
    Compute bootstrap CIs for all major metrics (MCC, F1, PR-AUC).

    Parameters
    ----------
    y_true : array-like
        True labels
    y_pred : array-like
        Predicted labels
    y_prob : array-like
        Predicted probabilities
    n_bootstrap : int
        Number of bootstrap iterations
    ci_level : float
        Confidence level
    random_state : int
        Random seed

    Returns
    -------
    dict with bootstrap CI results for each metric
    """
    results = {}

    # MCC
    def mcc_metric(y_t, y_p):
        return matthews_corrcoef(y_t, y_p)

    results['mcc'] = bootstrap_ci(
        y_true, y_pred, y_prob, mcc_metric,
        n_bootstrap=n_bootstrap, ci_level=ci_level, random_state=random_state
    )

    # F1 (positive class)
    def f1_metric(y_t, y_p):
        from sklearn.metrics import f1_score
        return f1_score(y_t, y_p, zero_division=0)

    results['f1'] = bootstrap_ci(
        y_true, y_pred, y_prob, f1_metric,
        n_bootstrap=n_bootstrap, ci_level=ci_level, random_state=random_state
    )

    # PR-AUC
    def prauc_metric(y_t, y_p):
        return average_precision_score(y_t, y_p)

    results['pr_auc'] = bootstrap_ci(
        y_true, y_pred, y_prob, prauc_metric,
        n_bootstrap=n_bootstrap, ci_level=ci_level, random_state=random_state
    )

    # ROC-AUC
    def rocauc_metric(y_t, y_p):
        return roc_auc_score(y_t, y_p)

    results['roc_auc'] = bootstrap_ci(
        y_true, y_pred, y_prob, rocauc_metric,
        n_bootstrap=n_bootstrap, ci_level=ci_level, random_state=random_state
    )

    return results


def get_metrics(y_test, y_pred_proba, feature_importance=None, compute_bootstrap_ci=False, n_bootstrap=1000):
    if len(y_pred_proba.shape) > 1:
        y_pred = (y_pred_proba[:, 1] >= 0.5).astype(int)
        y_prob = y_pred_proba[:, 1]
    else:
        y_pred = (y_pred_proba >= 0.5).astype(int) # Assuming binary output if not proba
        y_prob = y_pred_proba

    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    # Return NaN instead of 0 for single-class case (AUC is undefined, not "bad")
    if y_prob is not None and len(np.unique(y_test)) > 1:
        auc_score_val = roc_auc_score(y_test, y_prob)
        # PR-AUC (Precision-Recall AUC) - more informative for imbalanced data
        pr_auc_score_val = average_precision_score(y_test, y_prob)
    else:
        auc_score_val = np.nan  # AUC is undefined for single-class, not 0
        pr_auc_score_val = np.nan

    # Calculate MCC (Matthews Correlation Coefficient)
    try:
        mcc_score = matthews_corrcoef(y_test, y_pred)
    except Exception:
        mcc_score = np.nan

    # Class distribution in test set
    n_total = len(y_test)
    n_positive = int(np.sum(y_test))
    n_negative = n_total - n_positive
    class_distribution = {
        'n_total': n_total,
        'n_positive': n_positive,
        'n_negative': n_negative,
        'positive_ratio': n_positive / n_total if n_total > 0 else 0
    }

    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision_0': report.get('0', {}).get('precision', 0),
        'recall_0': report.get('0', {}).get('recall', 0),
        'f1_0': report.get('0', {}).get('f1-score', 0),
        'precision_1': report.get('1', {}).get('precision', 0),
        'recall_1': report.get('1', {}).get('recall', 0),
        'f1_1': report.get('1', {}).get('f1-score', 0),
        'auc': auc_score_val,
        'pr_auc': pr_auc_score_val,  # Precision-Recall AUC
        'mcc': mcc_score,  # Matthews Correlation Coefficient
        'class_distribution': class_distribution,  # Class distribution in test fold
        # ADD y_test and y_prob here for ROC data collection
        'y_test_fold': y_test.tolist() if hasattr(y_test, 'tolist') else list(y_test),
        'y_prob_fold': y_prob.tolist() if hasattr(y_prob, 'tolist') else list(y_prob)
    }

    # Compute bootstrap confidence intervals if requested
    if compute_bootstrap_ci and len(np.unique(y_test)) > 1:
        try:
            bootstrap_cis = compute_all_bootstrap_cis(
                y_test, y_pred, y_prob,
                n_bootstrap=n_bootstrap,
                ci_level=0.95,
                random_state=42
            )
            metrics['bootstrap_ci'] = bootstrap_cis
        except Exception as e:
            logging.warning(f"Bootstrap CI computation failed: {e}")
            metrics['bootstrap_ci'] = None
    else:
        metrics['bootstrap_ci'] = None

    if feature_importance is not None:
        # Store as a list of floats, not a string
        try:
            metrics['feature_importance'] = [float(val) for val in feature_importance]
        except (TypeError, ValueError) as e:
            logging.warning(f"Could not convert feature_importance to list of floats: {e}. Storing as None.")
            metrics['feature_importance'] = None
    else:
        metrics['feature_importance'] = None
    return metrics

def plot_roc_curves(classifiers_data, project_name, level, output_dir):
    plt.figure(figsize=(10, 8))
    for name, data_folds in classifiers_data.items():
        if not data_folds: continue
        mean_fpr = np.linspace(0, 1, 100)
        tprs, aucs_fold = [], []
        for fold_data in data_folds:
            if 'y_test' not in fold_data or 'y_prob' not in fold_data: continue
            fpr, tpr, _ = roc_curve(fold_data['y_test'], fold_data['y_prob'])
            interp_tpr = np.interp(mean_fpr, fpr, tpr)
            interp_tpr[0] = 0.0
            tprs.append(interp_tpr)
            aucs_fold.append(auc(fpr, tpr))
        if not tprs: continue
        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = np.mean(aucs_fold)
        std_auc = np.std(aucs_fold)
        plt.plot(mean_fpr, mean_tpr, label=f'{name} (AUC = {mean_auc:.2f} ± {std_auc:.2f})')

    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curves - {project_name} ({level} level)')
    plt.legend(loc='lower right')
    plt.grid(True)
    roc_path = output_dir / 'roc_curves.png'
    plt.savefig(roc_path)
    plt.close()
    logging.info(f"ROC curves plot saved to {roc_path}")

# Helper function for JSON serialization
def convert_numpy_to_list_recursive(data):
    """Recursively converts numpy arrays, pandas Series/DataFrames, and numpy scalar types
    in a nested data structure to Python lists, dicts, and native scalar types."""
    if isinstance(data, pd.Series):
        return data.tolist()
    if isinstance(data, np.ndarray):
        return data.tolist()
    if isinstance(data, pd.DataFrame): # Convert DataFrame to a list of records or dict of lists
        return data.to_dict(orient='list')
    if isinstance(data, dict):
        return {k: convert_numpy_to_list_recursive(v) for k, v in data.items()}
    if isinstance(data, list):
        return [convert_numpy_to_list_recursive(item) for item in data]
    # Handle numpy scalar types
    if isinstance(data, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64,
                        np.uint8, np.uint16, np.uint32, np.uint64)):
        return int(data)
    if isinstance(data, (np.float64, np.float16, np.float32, np.longdouble)): # Changed np.float_ to np.float64
        # Handle NaN values - convert to None for JSON compatibility
        if np.isnan(data):
            return None
        return float(data)
    if isinstance(data, np.bool_):
        return bool(data)
    if isinstance(data, np.str_): # Handle numpy strings
        return str(data)
    # Handle Python float NaN
    if isinstance(data, float) and np.isnan(data):
        return None
    return data

def plot_scores_barchart(results_data, score_key, plot_title, file_name, project_name, level, output_dir):
    """Generates and saves a bar chart for specified scores."""
    model_names = []
    scores = []
    # Extract model names and their corresponding scores
    for model_name, metrics_dict in results_data.items():
        if isinstance(metrics_dict, dict) and score_key in metrics_dict:
            # Prettify model names for display (e.g., 'random_forest' -> 'Random Forest')
            model_names.append(model_name.replace('_', ' ').title())
            scores.append(metrics_dict[score_key])

    if not model_names:
        logging.warning(f"No data to plot for {plot_title} for {project_name} ({level}). Skipping.")
        return

    # Dynamically adjust figure width based on the number of models
    plt.figure(figsize=(max(10, len(model_names) * 0.8), 7)) # Increased height slightly for better layout

    # Use a color map for bars
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(model_names))) # Adjusted color range
    bars = plt.bar(model_names, scores, color=colors)

    plt.xlabel('Machine Learning Model', fontsize=12)
    # Create a clean y-axis label (e.g., 'f1_1' -> 'F1 Score', 'accuracy' -> 'Accuracy')
    y_label = score_key.replace('_1', '').replace('_', ' ').title()
    if 'F1' in y_label: # Ensure 'F1 Score' instead of 'F1'
        y_label = 'F1 Score'
    plt.ylabel(y_label, fontsize=12)

    plt.title(f'{plot_title} - {project_name} ({level.title()} Level, Resampling: {results_data.get("resampling", "N/A")})', fontsize=14)
    plt.xticks(rotation=45, ha="right", fontsize=10)
    plt.yticks(np.arange(0, 1.1, 0.1), fontsize=10) # Y-axis from 0 to 1.0
    plt.ylim(0, 1.05) # Ensure plot limit is just above 1.0
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Add score values on top of bars for better readability
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.015, f'{yval:.3f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout(pad=1.5) # Add some padding

    plot_path = output_dir / f"{file_name}.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    logging.info(f"{plot_title} plot saved to {plot_path}")

class MetricsWithModel:
    def __init__(self, metrics_dict, model):
        self.metrics = metrics_dict
        self.model = model
        for key, value in metrics_dict.items():
            setattr(self, key, value)

# --- Optimization Functions (Simplified) ---
def save_optimization_results(project_name, level, resampling_method, model_name, best_params, best_score, output_dir, test_metrics=None):
    """Save optimization results to a JSON file."""
    results = {
        'project': project_name,
        'level': level,
        'resampling_method': resampling_method,
        'model': model_name,
        'best_parameters': best_params,
        'best_cv_score': best_score,  # Cross-validation score from GridSearchCV
        'timestamp': datetime.datetime.now().isoformat()
    }

    # Add test set performance metrics if provided
    if test_metrics:
        results['test_performance'] = {
            'accuracy': test_metrics.get('accuracy', None),
            'f1_score': test_metrics.get('f1_1', None),  # F1 score for positive class
            'precision': test_metrics.get('precision_1', None),
            'recall': test_metrics.get('recall_1', None),
            'auc': test_metrics.get('auc', None),
            'f1_0': test_metrics.get('f1_0', None),  # F1 score for negative class
            'precision_0': test_metrics.get('precision_0', None),
            'recall_0': test_metrics.get('recall_0', None)
        }

    # Create optimization results directory
    optimization_dir = output_dir / 'optimization_results'
    optimization_dir.mkdir(parents=True, exist_ok=True)

    # Save to individual file
    filename = f"{project_name}_{level}_{resampling_method}_{model_name}_optimization.json"
    filepath = optimization_dir / filename

    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        logging.info(f"Optimization results saved to {filepath}")
    except Exception as e:
        logging.error(f"Error saving optimization results to {filepath}: {e}")

    # Also append to a combined file for all optimizations
    combined_file = optimization_dir / f"{project_name}_{level}_{resampling_method}_all_optimizations.json"

    try:
        # Load existing results if file exists
        all_results = []
        if combined_file.exists():
            with open(combined_file, 'r', encoding='utf-8') as f:
                all_results = json.load(f)

        # Add new result
        all_results.append(results)

        # Save back
        with open(combined_file, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        logging.info(f"Optimization results appended to {combined_file}")
    except Exception as e:
        logging.error(f"Error updating combined optimization results file {combined_file}: {e}")

def _optimize_model_optuna(model_name, X_train, y_train, groups=None, timestamps=None,
                           resampling_strategy=None, class_weight=None, output_dir=None):
    """
    Optimize model hyperparameters using Optuna-TPE.

    New unified protocol:
    - Optimizer: Optuna with TPE sampler
    - Budget: 100 trials per model
    - Metric: MCC (Matthews Correlation Coefficient)
    - Seed: 42 (fixed for reproducibility)
    - Early stopping: Enabled for boosting models

    Parameters
    ----------
    model_name : str
        Name of the model to tune
    X_train : array-like
        Training features
    y_train : array-like
        Training labels
    groups : array-like, optional
        Group labels (commit sha) for temporal inner CV
    timestamps : array-like, optional
        Timestamps for temporal ordering
    resampling_strategy : str, optional
        Resampling strategy name (if None, no resampling)
    class_weight : str, optional
        'balanced' or None for class weighting
    output_dir : Path, optional
        Directory to save tuning results

    Returns
    -------
    model : fitted model with best parameters
    best_params : dict of best parameters
    best_score : float, best MCC score
    """
    logging.info(f"Optimizing {model_name} with Optuna-TPE (100 trials, MCC metric)...")

    # Get resampler if specified
    resampler = None
    if resampling_strategy and resampling_strategy != 'none':
        resampler = get_resampling_method(resampling_strategy)

    # Create tuner
    config = TuningConfig(
        n_trials=OPTUNA_N_TRIALS,
        random_seed=OPTUNA_RANDOM_SEED,
        optimization_metric='mcc'
    )
    tuner = OptunaHyperparameterTuner(config=config, output_dir=output_dir)

    try:
        model, best_params, best_score = tuner.tune_model(
            model_name=model_name,
            X_train=np.asarray(X_train),
            y_train=np.asarray(y_train),
            groups=groups,
            timestamps=timestamps,
            resampler=resampler,
            class_weight=class_weight
        )

        logging.info(f"Best MCC for {model_name}: {best_score:.4f}")
        logging.info(f"Best parameters: {best_params}")

        return model, best_params, best_score

    except Exception as e:
        logging.error(f"Optuna tuning failed for {model_name}: {e}. Returning default model.")
        # Create default model
        default_model = _create_default_model(model_name, class_weight)

        # Apply resampling if specified
        X_fit, y_fit = np.asarray(X_train), np.asarray(y_train)
        if resampler is not None:
            try:
                X_fit, y_fit = resampler.fit_resample(X_fit, y_fit)
            except Exception:
                pass

        default_model.fit(X_fit, y_fit)
        return default_model, {}, 0.0


def _create_default_model(model_name, class_weight=None):
    """Create a model with default parameters."""
    if model_name == 'xgboost':
        return xgb.XGBClassifier(
            objective='binary:logistic', use_label_encoder=False,
            eval_metric='logloss', random_state=42
        )
    elif model_name == 'lightgbm':
        cw = 'balanced' if class_weight == 'balanced' else None
        return lgb.LGBMClassifier(class_weight=cw, verbose=-1, random_state=42)
    elif model_name == 'catboost':
        acw = 'Balanced' if class_weight == 'balanced' else None
        return CatBoostClassifier(verbose=0, auto_class_weights=acw, random_state=42)
    elif model_name == 'random_forest':
        cw = 'balanced' if class_weight == 'balanced' else None
        return RandomForestClassifier(class_weight=cw, random_state=42)
    elif model_name == 'logistic_regression':
        cw = 'balanced' if class_weight == 'balanced' else None
        return LogisticRegression(class_weight=cw, random_state=42, max_iter=2000)
    elif model_name == 'gradient_boosting':
        return GradientBoostingClassifier(random_state=42)
    elif model_name == 'decision_tree':
        cw = 'balanced' if class_weight == 'balanced' else None
        return DecisionTreeClassifier(class_weight=cw, random_state=42)
    elif model_name == 'mlp':
        return MLPClassifier(random_state=42, max_iter=2000, early_stopping=True)
    elif model_name == 'naive_bayes':
        return GaussianNB()
    else:
        raise ValueError(f"Unknown model: {model_name}")


# Legacy _optimize_model for backward compatibility (can be removed later)
def _optimize_model(model, param_grid, X_train, y_train, model_name, inner_cv_folds=3,
                    groups=None, timestamps=None, resampling_strategy=None, use_scaler=False):
    """
    [DEPRECATED] Legacy GridSearchCV-based optimization.
    Use _optimize_model_optuna instead for the new unified protocol.
    """
    logging.warning(f"Using legacy GridSearchCV for {model_name}. Consider switching to Optuna-based tuning.")

    # Determine actual CV folds based on data size
    actual_cv_folds = min(inner_cv_folds, len(y_train) // 10)
    actual_cv_folds = max(2, actual_cv_folds)

    # Create temporal-aware inner CV
    if groups is not None and len(groups) == len(y_train):
        inner_cv = InnerTemporalCV(n_splits=actual_cv_folds, groups=groups, timestamps=timestamps)
    else:
        inner_cv = TimeSeriesSplit(n_splits=actual_cv_folds)

    # Build pipeline if resampling is needed
    if resampling_strategy and resampling_strategy != 'none':
        resampler = get_resampling_method(resampling_strategy)
        if resampler is not None:
            pipeline_steps = []
            if use_scaler:
                pipeline_steps.append(('scaler', StandardScaler()))
            pipeline_steps.append(('resampler', resampler))
            pipeline_steps.append(('classifier', model))
            pipeline = ImbPipeline(pipeline_steps)

            adjusted_param_grid = {}
            if isinstance(param_grid, list):
                adjusted_param_grid = []
                for pg in param_grid:
                    adjusted_pg = {f'classifier__{k}': v for k, v in pg.items()}
                    adjusted_param_grid.append(adjusted_pg)
            else:
                adjusted_param_grid = {f'classifier__{k}': v for k, v in param_grid.items()}

            grid_search = GridSearchCV(
                pipeline, adjusted_param_grid, cv=inner_cv, scoring='f1', n_jobs=-1, verbose=0
            )
        else:
            grid_search = GridSearchCV(model, param_grid, cv=inner_cv, scoring='f1', n_jobs=-1, verbose=0)
    else:
        grid_search = GridSearchCV(model, param_grid, cv=inner_cv, scoring='f1', n_jobs=-1, verbose=0)

    try:
        grid_search.fit(X_train, y_train)
        best_estimator = grid_search.best_estimator_
        if hasattr(best_estimator, 'named_steps') and 'classifier' in best_estimator.named_steps:
            best_model = best_estimator.named_steps['classifier']
            best_params = {k.replace('classifier__', ''): v for k, v in grid_search.best_params_.items()}
        else:
            best_model = best_estimator
            best_params = grid_search.best_params_
        return best_model, best_params, grid_search.best_score_
    except Exception as e:
        logging.error(f"GridSearchCV failed for {model_name}: {e}. Returning default model.")
        model.fit(X_train, y_train)
        return model, {}, 0.0


# =============================================================================
# Optuna-based Optimization Functions (New Unified Protocol)
# =============================================================================
# Protocol: Optuna-TPE, 100 trials, MCC metric, seed=42
# Tuned: xgboost, lightgbm, catboost, random_forest, logistic_regression,
#        gradient_boosting, decision_tree, mlp, stacking (meta-learner only)
# Not tuned: naive_bayes (default), voting (fixed soft voting)
# =============================================================================

def optimize_naive_bayes(X_train, y_train, groups=None, timestamps=None, resampling_strategy=None):
    """
    Create Naive Bayes with default parameters (NOT tuned).

    Rationale: GaussianNB has very few tunable parameters (var_smoothing),
    and default values generally work well. This keeps the protocol simple.
    """
    logging.info("Creating Naive Bayes with default parameters (not tuned)")

    model = GaussianNB()

    # Apply resampling if specified
    resampler = None
    if resampling_strategy and resampling_strategy != 'none':
        resampler = get_resampling_method(resampling_strategy)

    X_fit, y_fit = np.asarray(X_train), np.asarray(y_train)
    if resampler is not None:
        try:
            X_fit, y_fit = resampler.fit_resample(X_fit, y_fit)
        except Exception as e:
            logging.warning(f"Resampling failed for Naive Bayes: {e}")

    model.fit(X_fit, y_fit)
    return model, {'tuned': False, 'reason': 'default_parameters'}, None


def optimize_xgboost(X_train, y_train, groups=None, timestamps=None, resampling_strategy=None):
    """
    Tune XGBoost with Optuna-TPE (100 trials, MCC metric).
    Early stopping: 50 rounds on inner validation.
    """
    class_weight = get_class_weight_value(resampling_strategy)
    return _optimize_model_optuna(
        model_name='xgboost',
        X_train=X_train, y_train=y_train,
        groups=groups, timestamps=timestamps,
        resampling_strategy=resampling_strategy,
        class_weight=class_weight
    )


def optimize_random_forest(X_train, y_train, groups=None, timestamps=None, resampling_strategy=None):
    """
    Tune Random Forest with Optuna-TPE (100 trials, MCC metric).
    """
    class_weight = get_class_weight_value(resampling_strategy)
    return _optimize_model_optuna(
        model_name='random_forest',
        X_train=X_train, y_train=y_train,
        groups=groups, timestamps=timestamps,
        resampling_strategy=resampling_strategy,
        class_weight=class_weight
    )


def optimize_lightgbm(X_train, y_train, groups=None, timestamps=None, resampling_strategy=None):
    """
    Tune LightGBM with Optuna-TPE (100 trials, MCC metric).
    Early stopping: 50 rounds on inner validation.
    """
    class_weight = get_class_weight_value(resampling_strategy)
    return _optimize_model_optuna(
        model_name='lightgbm',
        X_train=X_train, y_train=y_train,
        groups=groups, timestamps=timestamps,
        resampling_strategy=resampling_strategy,
        class_weight=class_weight
    )


def optimize_catboost(X_train, y_train, groups=None, timestamps=None, resampling_strategy=None):
    """
    Tune CatBoost with Optuna-TPE (100 trials, MCC metric).
    Early stopping: od_wait=50 on inner validation.
    """
    class_weight = get_class_weight_value(resampling_strategy)
    return _optimize_model_optuna(
        model_name='catboost',
        X_train=X_train, y_train=y_train,
        groups=groups, timestamps=timestamps,
        resampling_strategy=resampling_strategy,
        class_weight=class_weight
    )


def optimize_logistic_regression(X_train, y_train, groups=None, timestamps=None, resampling_strategy=None):
    """
    Tune Logistic Regression with Optuna-TPE (100 trials, MCC metric).
    Solver is auto-selected based on penalty (l1 -> liblinear, l2 -> lbfgs).
    """
    class_weight = get_class_weight_value(resampling_strategy)
    return _optimize_model_optuna(
        model_name='logistic_regression',
        X_train=X_train, y_train=y_train,
        groups=groups, timestamps=timestamps,
        resampling_strategy=resampling_strategy,
        class_weight=class_weight
    )


def optimize_gradient_boosting(X_train, y_train, groups=None, timestamps=None, resampling_strategy=None):
    """
    Tune Gradient Boosting (sklearn) with Optuna-TPE (100 trials, MCC metric).
    """
    class_weight = get_class_weight_value(resampling_strategy)
    return _optimize_model_optuna(
        model_name='gradient_boosting',
        X_train=X_train, y_train=y_train,
        groups=groups, timestamps=timestamps,
        resampling_strategy=resampling_strategy,
        class_weight=class_weight
    )


def optimize_decision_tree(X_train, y_train, groups=None, timestamps=None, resampling_strategy=None):
    """
    Tune Decision Tree with Optuna-TPE (100 trials, MCC metric).
    """
    class_weight = get_class_weight_value(resampling_strategy)
    return _optimize_model_optuna(
        model_name='decision_tree',
        X_train=X_train, y_train=y_train,
        groups=groups, timestamps=timestamps,
        resampling_strategy=resampling_strategy,
        class_weight=class_weight
    )


def optimize_mlp(X_train, y_train, groups=None, timestamps=None, resampling_strategy=None):
    """
    Tune MLP with Optuna-TPE (100 trials, MCC metric).
    Early stopping: n_iter_no_change=20.
    """
    class_weight = get_class_weight_value(resampling_strategy)
    return _optimize_model_optuna(
        model_name='mlp',
        X_train=X_train, y_train=y_train,
        groups=groups, timestamps=timestamps,
        resampling_strategy=resampling_strategy,
        class_weight=class_weight
    )


def optimize_voting(X_train, y_train, groups=None, timestamps=None, resampling_strategy=None,
                    tuned_estimators=None):
    """
    Create VotingClassifier with fixed configuration (NOT tuned).

    Configuration:
    - Base estimators: [LR, RF, XGB, LGBM, CatBoost] (should be tuned versions)
    - Voting: soft (using probabilities)
    - Weights: uniform (equal)

    If tuned_estimators is not provided, creates new models with default params.
    """
    logging.info("Creating VotingClassifier with fixed soft voting (not tuned)")

    class_weight = get_class_weight_value(resampling_strategy)
    cw = 'balanced' if class_weight == 'balanced' else None

    if tuned_estimators is not None:
        # Use provided tuned estimators
        estimators = []
        for name in ['logistic_regression', 'random_forest', 'xgboost', 'lightgbm', 'catboost']:
            if name in tuned_estimators:
                estimators.append((name, clone(tuned_estimators[name])))
    else:
        # Create default estimators
        estimators = [
            ('logistic_regression', LogisticRegression(class_weight=cw, random_state=42, max_iter=2000)),
            ('random_forest', RandomForestClassifier(class_weight=cw, random_state=42)),
            ('xgboost', xgb.XGBClassifier(objective='binary:logistic', use_label_encoder=False,
                                          eval_metric='logloss', random_state=42)),
            ('lightgbm', lgb.LGBMClassifier(class_weight=cw, verbose=-1, random_state=42)),
            ('catboost', CatBoostClassifier(verbose=0, auto_class_weights='Balanced' if cw else None,
                                           random_state=42))
        ]

    model = VotingClassifier(
        estimators=estimators,
        voting='soft',
        weights=None,  # Equal weights
        n_jobs=1
    )

    # Apply resampling if specified
    resampler = None
    if resampling_strategy and resampling_strategy != 'none':
        resampler = get_resampling_method(resampling_strategy)

    X_fit, y_fit = np.asarray(X_train), np.asarray(y_train)
    if resampler is not None:
        try:
            X_fit, y_fit = resampler.fit_resample(X_fit, y_fit)
        except Exception as e:
            logging.warning(f"Resampling failed for VotingClassifier: {e}")

    model.fit(X_fit, y_fit)

    return model, {'voting': 'soft', 'weights': 'uniform', 'tuned': False}, None


def optimize_stacking(X_train, y_train, groups=None, timestamps=None, resampling_strategy=None,
                      tuned_estimators=None):
    """
    Tune StackingClassifier meta-learner with Optuna-TPE (100 trials, MCC metric).

    Only tunes:
    - passthrough: {True, False}
    - final_estimator C: log-uniform [1e-4, 100]

    Base estimators should be tuned versions (not re-tuned inside stacking).
    """
    logging.info("Tuning StackingClassifier meta-learner with Optuna-TPE")

    class_weight = get_class_weight_value(resampling_strategy)
    cw = 'balanced' if class_weight == 'balanced' else None

    # Prepare base estimators
    if tuned_estimators is not None:
        base_estimators = []
        for name in ['logistic_regression', 'random_forest', 'xgboost', 'lightgbm', 'catboost']:
            if name in tuned_estimators:
                base_estimators.append((name, clone(tuned_estimators[name])))
    else:
        # Create default base estimators
        base_estimators = [
            ('logistic_regression', LogisticRegression(class_weight=cw, random_state=42, max_iter=2000)),
            ('random_forest', RandomForestClassifier(class_weight=cw, random_state=42)),
            ('xgboost', xgb.XGBClassifier(objective='binary:logistic', use_label_encoder=False,
                                          eval_metric='logloss', random_state=42)),
            ('lightgbm', lgb.LGBMClassifier(class_weight=cw, verbose=-1, random_state=42)),
            ('catboost', CatBoostClassifier(verbose=0, auto_class_weights='Balanced' if cw else None,
                                           random_state=42))
        ]

    if len(base_estimators) < 2:
        logging.warning("StackingClassifier requires at least 2 base estimators. Using defaults.")
        base_estimators = [
            ('rf', RandomForestClassifier(class_weight=cw, random_state=42)),
            ('xgb', xgb.XGBClassifier(objective='binary:logistic', use_label_encoder=False,
                                      eval_metric='logloss', random_state=42))
        ]

    # Get resampler if specified
    resampler = None
    if resampling_strategy and resampling_strategy != 'none':
        resampler = get_resampling_method(resampling_strategy)

    # Create tuner for stacking meta-learner
    config = TuningConfig(
        n_trials=OPTUNA_N_TRIALS,
        random_seed=OPTUNA_RANDOM_SEED,
        optimization_metric='mcc'
    )
    tuner = OptunaHyperparameterTuner(config=config)

    try:
        model, params, score = tuner.tune_stacking(
            X_train=np.asarray(X_train),
            y_train=np.asarray(y_train),
            base_estimators=base_estimators,
            groups=groups,
            timestamps=timestamps,
            resampler=resampler,
            class_weight=class_weight
        )

        logging.info(f"Best MCC for Stacking: {score:.4f}")
        logging.info(f"Best meta-learner params: {params}")

        return model, params, score

    except Exception as e:
        logging.error(f"Optuna tuning failed for Stacking: {e}. Returning default model.")

        # Create default stacking model
        final_estimator = LogisticRegression(class_weight=cw, random_state=42, max_iter=2000)
        model = StackingClassifier(
            estimators=base_estimators,
            final_estimator=final_estimator,
            passthrough=False,
            cv=3,
            n_jobs=1
        )

        X_fit, y_fit = np.asarray(X_train), np.asarray(y_train)
        if resampler is not None:
            try:
                X_fit, y_fit = resampler.fit_resample(X_fit, y_fit)
            except Exception:
                pass

        model.fit(X_fit, y_fit)
        return model, {'passthrough': False}, 0.0


# --- Analysis Functions ---
def _run_analysis(model_name, model_func, X_train, X_test, y_train, y_test, project_name, output_dir, optimize=False, needs_scaling=False, level=None, resampling_method=None):
    logging.info(f"Running analysis with {model_name} for {project_name}...")

    X_train_processed, X_test_processed = X_train.copy(), X_test.copy()

    if needs_scaling:
        scaler = StandardScaler()
        X_train_processed = pd.DataFrame(scaler.fit_transform(X_train_processed), columns=X_train.columns, index=X_train.index)
        X_test_processed = pd.DataFrame(scaler.transform(X_test_processed), columns=X_train.columns, index=X_test.index)

    # Check for single class in training data before fitting
    unique_classes_in_train = np.unique(y_train)
    if len(unique_classes_in_train) < 2:
        logging.warning(
            f"Skipping model fitting for {model_name} on {project_name} (Level: {level}, Resampling: {resampling_method}, Fold being processed). "
            f"Training data for this fold contains only one class: {unique_classes_in_train}. "
            f"X_train shape: {X_train_processed.shape}, y_train counts: {pd.Series(y_train).value_counts().to_dict()}"
        )
        # Return default/error metrics as the model cannot be trained
        error_metrics = {
            'accuracy': 0.0, 'precision_0': 0.0, 'recall_0': 0.0, 'f1_0': 0.0,
            'precision_1': 0.0, 'recall_1': 0.0, 'f1_1': 0.0, 'auc': 0.0, # Using 0.0 for AUC, could be 0.5 if interpreted as random
            'y_test_fold': y_test.tolist() if hasattr(y_test, 'tolist') else list(y_test),
            # Provide a non-informative probability for y_prob_fold to avoid downstream errors in ROC plotting etc.
            # Assuming y_test is not empty. If y_test could be empty, this needs another check.
            'y_prob_fold': [0.5] * len(y_test) if len(y_test) > 0 else [],
            'feature_importance': None,
            'error_message': f"Training data for this fold only contained class(es): {unique_classes_in_train}"
        }
        # For non-optimized path, 'model' instance is created before this check.
        # For optimized path, model_func would be the optimizer.
        # Returning None for the model part of MetricsWithModel when fitting is skipped.
        return MetricsWithModel(error_metrics, model=None)

    if optimize:
        # model_func here is the optimization function that returns (model, params, score)
        result = model_func(X_train_processed, y_train)
        if isinstance(result, tuple) and len(result) == 3:
            model, best_params, best_score = result
            # Note: save_optimization_results will be called later with test metrics
        else:
            # Fallback for old style optimization functions
            model = result
            best_params, best_score = {}, 0.0
    else:
        # This assumes model_func is a constructor if not optimizing
        # We need to adjust this logic or ensure model_func is always the constructor
        # For simplicity, direct instantiation for non-optimized path
        # Use global class weight mode to determine class_weight parameter
        class_weight_value = get_class_weight_value(resampling_method)
        # CatBoost uses 'Balanced' string instead of 'balanced'
        catboost_class_weight = 'Balanced' if class_weight_value == 'balanced' else None

        if model_name == 'naive_bayes': model = GaussianNB()
        elif model_name == 'xgboost': model = xgb.XGBClassifier(objective='binary:logistic', use_label_encoder=False, eval_metric='logloss', random_state=42)
        elif model_name == 'random_forest':
            model = RandomForestClassifier(class_weight=class_weight_value, random_state=42)
        elif model_name == 'logistic_regression':
            model = LogisticRegression(class_weight=class_weight_value, random_state=42, max_iter=1000)
        elif model_name == 'catboost':
            model = CatBoostClassifier(verbose=0, auto_class_weights=catboost_class_weight, random_state=42)
        elif model_name == 'lightgbm':
            model = lgb.LGBMClassifier(class_weight=class_weight_value, verbose=-1, random_state=42)
        elif model_name == 'gradient_boosting': model = GradientBoostingClassifier(random_state=42)
        elif model_name == 'decision_tree':
            model = DecisionTreeClassifier(class_weight=class_weight_value, random_state=42)
        # Voting and Stacking require base estimators, handled separately or need specific non-optimized setup
        elif model_name == 'voting':
            estimators = [
                ('rf', RandomForestClassifier(n_estimators=50, random_state=42, class_weight=class_weight_value)),
                ('xgb', xgb.XGBClassifier(objective='binary:logistic', use_label_encoder=False, eval_metric='logloss', n_estimators=50, random_state=42))
            ]
            model = VotingClassifier(estimators=estimators, voting='soft')
        elif model_name == 'stacking':
            estimators = [
                ('rf', RandomForestClassifier(n_estimators=50, random_state=42, class_weight=class_weight_value)),
                ('xgb', xgb.XGBClassifier(objective='binary:logistic', use_label_encoder=False, eval_metric='logloss', n_estimators=50, random_state=42))
            ]
            final_estimator = LogisticRegression(class_weight=class_weight_value, random_state=42, max_iter=1000)
            model = StackingClassifier(estimators=estimators, final_estimator=final_estimator, cv=3)
        elif model_name == 'mlp':
            if len(y_train) < 20:
                logging.warning(f"MLP (non-optimized): Training data size ({len(y_train)}) is too small for early stopping.")
                model = MLPClassifier(hidden_layer_sizes=(50,), max_iter=2000, early_stopping=False, random_state=42)
            else:
                model = MLPClassifier(hidden_layer_sizes=(50,), max_iter=2000, early_stopping=True, validation_fraction=0.1, n_iter_no_change=10, random_state=42)
        else: raise ValueError(f"Unknown model: {model_name}")

        model.fit(X_train_processed, y_train)

    y_pred_proba_raw = model.predict_proba(X_test_processed)
    y_pred_proba_for_metrics = y_pred_proba_raw # Default to raw

    # Adjust y_pred_proba if it's degenerate (e.g., from single-class training)
    # to ensure it's (n_samples, 2) for binary classification passed to get_metrics.
    if hasattr(model, 'classes_'):
        n_classes_learned = len(model.classes_)
        if n_classes_learned == 1:
            logging.warning(
                f"Model {model_name} (project: {project_name}, level: {level}, resampling: {resampling_method}) "
                f"was trained on a single class: {model.classes_[0]}. Adjusting predict_proba output for metrics."
            )
            n_samples = y_pred_proba_raw.shape[0]
            y_pred_proba_adjusted = np.zeros((n_samples, 2)) # Assuming 2 target classes (0 and 1)

            # y_pred_proba_raw is (n_samples, 1), containing P(model.classes_[0])
            prob_of_single_learned_class = y_pred_proba_raw[:, 0]

            if model.classes_[0] == 0:
                y_pred_proba_adjusted[:, 0] = prob_of_single_learned_class  # P(class 0)
                y_pred_proba_adjusted[:, 1] = 1.0 - prob_of_single_learned_class # P(class 1)
            elif model.classes_[0] == 1:
                y_pred_proba_adjusted[:, 1] = prob_of_single_learned_class  # P(class 1)
                y_pred_proba_adjusted[:, 0] = 1.0 - prob_of_single_learned_class # P(class 0)
            else:
                # This case should ideally not happen if y is always 0 or 1.
                logging.error(f"Model {model_name} trained on single, unexpected class {model.classes_[0]}. Cannot reliably adjust probabilities for binary metrics.")
                # Fallback: keep raw, get_metrics might struggle or give poor results.
                pass # y_pred_proba_for_metrics remains y_pred_proba_raw

            y_pred_proba_for_metrics = y_pred_proba_adjusted

        elif n_classes_learned != 2 and y_pred_proba_raw.shape[1] != 2:
             # This condition handles cases where classes_ might be > 2 (multiclass model used mistakenly)
             # or if classes_ is 2 but shape is still not (N,2) for some reason.
            logging.warning(
                f"Model {model_name} (project: {project_name}, level: {level}, resampling: {resampling_method}) "
                f"has classes_ {model.classes_} and predict_proba shape {y_pred_proba_raw.shape}. "
                f"Expected 2 classes and (N,2) shape for binary metrics. Using raw probabilities."
            )
            # y_pred_proba_for_metrics remains y_pred_proba_raw; get_metrics will handle based on its shape.
    else:
        # Model does not have classes_ attribute, or it's None.
        # This might be a custom ensemble or something not exposing classes_ as expected.
        # We also check if the raw output is not the standard (N,2) for binary.
        if y_pred_proba_raw.ndim != 2 or y_pred_proba_raw.shape[1] != 2:
            logging.warning(
                f"Model {model_name} (project: {project_name}, level: {level}, resampling: {resampling_method}) "
                f"does not have 'classes_' attribute or its predict_proba output is not (N,2) (shape: {y_pred_proba_raw.shape}). "
                f"Metrics might be based on potentially uncalibrated scores if 1D, or handled by shape in get_metrics."
            )
        # y_pred_proba_for_metrics remains y_pred_proba_raw

    feature_importance = getattr(model, 'feature_importances_', None)
    if feature_importance is None and hasattr(model, 'coef_') and model.coef_.ndim == 1 : # For Linear models like Logistic Regression
        feature_importance = model.coef_
    elif feature_importance is None and hasattr(model, 'coef_') and model.coef_.ndim > 1: # For Linear models like Logistic Regression with multiclass
         feature_importance = model.coef_[0]

    metrics = get_metrics(y_test, y_pred_proba_for_metrics, feature_importance)

    # Add optimization results to the metrics dict to be processed later
    if optimize and isinstance(result, tuple) and len(result) == 3:
        model, best_params, best_score = result
        metrics['best_params'] = best_params
        metrics['best_cv_score'] = best_score

    logging.debug(f"{model_name} metrics for {project_name}: {metrics}")
    return MetricsWithModel(metrics, model)


def analyze_with_naive_bayes(X_train, X_test, y_train, y_test, project_name, output_dir, optimize=False, level=None, resampling_method=None):
    model_func = optimize_naive_bayes if optimize else GaussianNB
    return _run_analysis("naive_bayes", model_func, X_train, X_test, y_train, y_test, project_name, output_dir, optimize, needs_scaling=False, level=level, resampling_method=resampling_method)

def analyze_with_xgboost(X_train, X_test, y_train, y_test, project_name, output_dir, optimize=False, level=None, resampling_method=None):
    model_func = optimize_xgboost if optimize else lambda x,y: xgb.XGBClassifier(objective='binary:logistic', use_label_encoder=False, eval_metric='logloss', random_state=42).fit(x,y)
    return _run_analysis("xgboost", model_func, X_train, X_test, y_train, y_test, project_name, output_dir, optimize, needs_scaling=False, level=level, resampling_method=resampling_method)

def analyze_with_random_forest(X_train, X_test, y_train, y_test, project_name, output_dir, optimize=False, level=None, resampling_method=None):
    model_func = optimize_random_forest if optimize else lambda x,y: RandomForestClassifier(class_weight='balanced', random_state=42).fit(x,y)
    return _run_analysis("random_forest", model_func, X_train, X_test, y_train, y_test, project_name, output_dir, optimize, needs_scaling=False, level=level, resampling_method=resampling_method)

def analyze_with_logistic_regression(X_train, X_test, y_train, y_test, project_name, output_dir, optimize=False, level=None, resampling_method=None):
    model_func = optimize_logistic_regression if optimize else lambda x,y: LogisticRegression(class_weight='balanced', random_state=42, max_iter=1000).fit(x,y)
    return _run_analysis("logistic_regression", model_func, X_train, X_test, y_train, y_test, project_name, output_dir, optimize, needs_scaling=True, level=level, resampling_method=resampling_method)

def analyze_with_catboost(X_train, X_test, y_train, y_test, project_name, output_dir, optimize=False, level=None, resampling_method=None):
    model_func = optimize_catboost if optimize else lambda x,y: CatBoostClassifier(verbose=0, auto_class_weights='Balanced', random_state=42).fit(x,y)
    return _run_analysis("catboost", model_func, X_train, X_test, y_train, y_test, project_name, output_dir, optimize, needs_scaling=False, level=level, resampling_method=resampling_method)

def analyze_with_lightgbm(X_train, X_test, y_train, y_test, project_name, output_dir, optimize=False, level=None, resampling_method=None):
    model_func = optimize_lightgbm if optimize else lambda x,y: lgb.LGBMClassifier(class_weight='balanced', verbose=-1, random_state=42).fit(x,y)
    return _run_analysis("lightgbm", model_func, X_train, X_test, y_train, y_test, project_name, output_dir, optimize, needs_scaling=False, level=level, resampling_method=resampling_method)

def analyze_with_gradient_boosting(X_train, X_test, y_train, y_test, project_name, output_dir, optimize=False, level=None, resampling_method=None):
    model_func = optimize_gradient_boosting if optimize else lambda x,y: GradientBoostingClassifier(random_state=42).fit(x,y)
    return _run_analysis("gradient_boosting", model_func, X_train, X_test, y_train, y_test, project_name, output_dir, optimize, needs_scaling=False, level=level, resampling_method=resampling_method)

def analyze_with_decision_tree(X_train, X_test, y_train, y_test, project_name, output_dir, optimize=False, level=None, resampling_method=None):
    model_func = optimize_decision_tree if optimize else lambda x,y: DecisionTreeClassifier(class_weight='balanced', random_state=42).fit(x,y)
    return _run_analysis("decision_tree", model_func, X_train, X_test, y_train, y_test, project_name, output_dir, optimize, needs_scaling=False, level=level, resampling_method=resampling_method)

def analyze_with_voting(X_train, X_test, y_train, y_test, project_name, output_dir, optimize=False, level=None, resampling_method=None):
    if optimize:
        model_func = optimize_voting
    else:
        estimators = [
            ('rf', RandomForestClassifier(n_estimators=50, random_state=42, class_weight='balanced')),
            ('xgb', xgb.XGBClassifier(objective='binary:logistic', use_label_encoder=False, eval_metric='logloss', n_estimators=50, random_state=42))
        ]
        model_func = lambda x,y: VotingClassifier(estimators=estimators, voting='soft').fit(x,y)
    return _run_analysis("voting", model_func, X_train, X_test, y_train, y_test, project_name, output_dir, optimize, needs_scaling=False, level=level, resampling_method=resampling_method)

def analyze_with_mlp(X_train, X_test, y_train, y_test, project_name, output_dir, optimize=False, level=None, resampling_method=None):
    model_func = optimize_mlp if optimize else lambda x,y: MLPClassifier(hidden_layer_sizes=(50,), max_iter=2000, early_stopping=True, validation_fraction=0.1, n_iter_no_change=10, random_state=42).fit(x,y)
    return _run_analysis("mlp", model_func, X_train, X_test, y_train, y_test, project_name, output_dir, optimize, needs_scaling=True, level=level, resampling_method=resampling_method)

def analyze_with_stacking(X_train, X_test, y_train, y_test, project_name, output_dir, optimize=False, level=None, resampling_method=None):
    if optimize:
        model_func = optimize_stacking
    else:
        estimators = [
            ('rf', RandomForestClassifier(n_estimators=50, random_state=42, class_weight='balanced')),
            ('xgb', xgb.XGBClassifier(objective='binary:logistic', use_label_encoder=False, eval_metric='logloss', n_estimators=50, random_state=42))
        ]
        final_estimator = LogisticRegression(class_weight='balanced', random_state=42, max_iter=1000)
        model_func = lambda x,y: StackingClassifier(estimators=estimators, final_estimator=final_estimator, cv=3).fit(x,y)
    return _run_analysis("stacking", model_func, X_train, X_test, y_train, y_test, project_name, output_dir, optimize, needs_scaling=False, level=level, resampling_method=resampling_method)

def plot_feature_correlations(X, output_dir, project_name, level):
    if X.empty or X.shape[1] < 2:
        logging.warning(f"Not enough features for correlation plot for {project_name} ({level}). Skipping.")
        return
    plt.figure(figsize=(max(12, X.shape[1]*0.5), max(10, X.shape[1]*0.4))) # Dynamic sizing
    sns.heatmap(X.corr(), annot=True, cmap='coolwarm', center=0, fmt='.2f', square=False, annot_kws={"size": 8}) # Smaller annotations
    plt.title(f'Feature Correlations - {project_name} ({level} level)')
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    corr_path = output_dir / 'feature_correlations.png'
    plt.savefig(corr_path, dpi=300)
    plt.close()
    logging.info(f"Feature correlation plot saved to {corr_path}")


def plot_significance_heatmap(p_values_df, title, output_path):
    """
    Generates and saves a significance heatmap from post-hoc test p-values.

    This plot, using scikit-posthocs.sign_plot, visualizes a matrix of p-values
    from pairwise comparisons, indicating where statistically significant
    differences exist between groups.

    Args:
        p_values_df (pd.DataFrame): A square DataFrame of p-values from a post-hoc test.
                                    Column and index names should be the names of the models.
        title (str): The title for the plot.
        output_path (Path): The path to save the generated PNG file.
    """
    try:
        # The sign_plot function visualizes the p-value matrix from a pairwise comparison test.
        # It creates a heatmap where colors indicate significance level.
        # Adjusted figsize for better aspect ratio and to provide space for the colorbar.
        fig, ax = plt.subplots(figsize=(12, max(6, 0.7 * len(p_values_df.columns))), dpi=300)

        sp.sign_plot(p_values_df, ax=ax)

        ax.set_title(title, fontsize=14, pad=20)

        # NOTE: Using bbox_inches='tight' in savefig is more robust than fig.tight_layout()
        # for plots with colorbars created by external libraries like scikit-posthocs.
        # It ensures all artists (like the colorbar) are included in the saved figure.
        plt.savefig(output_path, bbox_inches='tight')
        plt.close(fig)
        logging.info(f"Significance heatmap (sign_plot) saved to {output_path}")
    except Exception as e:
        logging.error(f"Failed to generate significance heatmap at {output_path}: {e}", exc_info=True)


def analyze_project(project_name, level, resampling_strategy=None, n_folds=5, optimize=False, selected_features_config=None, methods_to_run=None, cli_args=None, progress=None, project_task_id=None, compute_bootstrap_ci=False, n_bootstrap=1000):
    """
    Analyze a project using temporal nested cross-validation.

    Implements time-aware 80/20 split with nested cross-validation:
    1. Data is sorted chronologically by commit_timestamp
    2. First 80% of commits used for training/validation (outer CV)
    3. Last 20% of commits used as final hold-out test
    4. Commit group integrity is maintained (all instances from same commit stay together)
    5. Standard deviation is computed across outer CV folds

    Args:
        compute_bootstrap_ci: If True, compute bootstrap confidence intervals for holdout metrics
        n_bootstrap: Number of bootstrap iterations (default: 1000)
    """
    logging.info(f"Starting analysis for project: {project_name}, level: {level}, resampling: {resampling_strategy}, optimize: {optimize}")

    # Check for overlap_only flag from CLI args
    overlap_only = cli_args.overlap_only if cli_args and hasattr(cli_args, 'overlap_only') else False

    df = load_project_data(project_name, level, sort_by_time=True, overlap_only=overlap_only)
    if df is None:
        logging.error(f"Failed to load data for {project_name} at level {level}. Skipping analysis.")
        return None

    # Extract temporal and grouping columns before feature preparation
    if 'sha' not in df.columns:
        logging.error(f"'sha' column not found in data for {project_name}. Cannot perform commit-based temporal splitting.")
        return None
    if 'commit_timestamp' not in df.columns:
        logging.error(f"'commit_timestamp' column not found in data for {project_name}. Cannot perform temporal splitting.")
        return None

    # Store sha and timestamp for splitting (before feature extraction)
    commit_groups = df['sha'].values
    commit_timestamps = df['commit_timestamp'].values

    X, y = prepare_features(df, level, selected_features_config, cli_args)
    if X is None or y is None or X.empty:
        logging.error(f"Failed to prepare features for {project_name} at level {level}. Skipping analysis.")
        return None

    if X.empty:
        logging.warning(f"No features to analyze for project {project_name}, level {level}. Skipping.")
        return None

    # This check needs to be before feature selection
    if len(y.unique()) < 2:
        logging.error(f"Project {project_name} at level {level} has only one class in the target variable. Skipping analysis as model cannot be trained.")
        return None

    print_class_distribution(f"Original Dataset ({project_name} - {level})", y)

    # --- Hierarchical Output Directory Structure ---
    # Structure: results_{level}_level/{project}/{cv_type}/{feature_set}/{resampling}/
    cv_type = get_cv_type_name(cli_args and hasattr(cli_args, 'shuffle_cv') and cli_args.shuffle_cv)
    feature_set = get_feature_set_name(cli_args and cli_args.exclude_go_metrics)
    resampling_name = resampling_strategy if resampling_strategy is not None else 'none'

    output_dir = get_analysis_output_dir(
        level=level,
        project_name=project_name,
        cv_type=cv_type,
        feature_set=feature_set,
        resampling=resampling_name
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    logging.info(f"Output directory: {output_dir}")
    logging.info(f"  CV Type: {cv_type}, Feature Set: {feature_set}, Resampling: {resampling_name}")

    # --- Temporal 80/20 Hold-out Split (commit-based) ---
    unique_commits_ordered = []
    seen_commits = set()
    for sha in commit_groups:
        if sha not in seen_commits:
            unique_commits_ordered.append(sha)
            seen_commits.add(sha)

    n_unique_commits = len(unique_commits_ordered)
    n_positive_samples = int(y.sum())

    logging.info(f"Dataset has {n_unique_commits} unique commits, {len(X)} total samples, {n_positive_samples} positive samples")

    # Split commits: 80% train/val, 20% hold-out test
    holdout_split_idx = int(n_unique_commits * 0.8)
    train_val_commits = set(unique_commits_ordered[:holdout_split_idx])
    holdout_commits = set(unique_commits_ordered[holdout_split_idx:])

    # Get sample indices for train/val and hold-out
    train_val_mask = np.isin(commit_groups, list(train_val_commits))
    holdout_mask = np.isin(commit_groups, list(holdout_commits))

    train_val_indices = np.where(train_val_mask)[0]
    holdout_indices = np.where(holdout_mask)[0]

    X_train_val = X.iloc[train_val_indices].reset_index(drop=True)
    y_train_val = y.iloc[train_val_indices].reset_index(drop=True)
    X_holdout = X.iloc[holdout_indices].reset_index(drop=True)
    y_holdout = y.iloc[holdout_indices].reset_index(drop=True)

    # Get groups and timestamps for train/val subset
    train_val_groups = commit_groups[train_val_indices]
    train_val_timestamps = commit_timestamps[train_val_indices]
    holdout_timestamps = commit_timestamps[holdout_indices]

    # Validate temporal integrity of hold-out split
    train_val_max_ts = np.max(train_val_timestamps)
    holdout_min_ts = np.min(holdout_timestamps)

    if train_val_max_ts >= holdout_min_ts:
        logging.error(
            f"CRITICAL: Temporal integrity violation in hold-out split! "
            f"train_val_max_timestamp ({train_val_max_ts}) >= holdout_min_timestamp ({holdout_min_ts})"
        )
    else:
        logging.info(
            f"Hold-out split validated: train/val ends at {train_val_max_ts}, "
            f"hold-out starts at {holdout_min_ts} (gap: {holdout_min_ts - train_val_max_ts}s)"
        )

    # Log split information
    logging.info(
        f"Temporal 80/20 split: {len(train_val_commits)} train/val commits ({len(X_train_val)} samples), "
        f"{len(holdout_commits)} hold-out commits ({len(X_holdout)} samples)"
    )
    print_class_distribution(f"Train/Val Set ({project_name})", y_train_val)
    print_class_distribution(f"Hold-out Test Set ({project_name})", y_holdout)

    # --- Assess Dataset Quality ---
    n_trainval_minority = int(y_train_val.sum())
    n_holdout_minority = int(y_holdout.sum())

    dataset_quality, quality_reasons = assess_dataset_quality(
        n_trainval=len(X_train_val),
        n_trainval_minority=n_trainval_minority,
        n_holdout=len(X_holdout),
        n_holdout_minority=n_holdout_minority
    )

    # Log dataset quality assessment
    if dataset_quality == DatasetQuality.PRIMARY:
        logging.info(f"Dataset quality: PRIMARY - suitable for main statistical analyses")
    elif dataset_quality == DatasetQuality.EXPLORATORY:
        logging.warning(
            f"Dataset quality: EXPLORATORY - results will be reported but excluded from "
            f"main statistical comparisons. Reasons: {quality_reasons['issues']}"
        )
    else:  # INSUFFICIENT
        logging.warning(
            f"Dataset quality: INSUFFICIENT - not included in primary statistical comparisons "
            f"due to limited minority samples. Reasons: {quality_reasons['issues']}. "
            f"Results will be reported for completeness but excluded from Friedman/Nemenyi tests."
        )
        # Still continue with analysis but mark results clearly

    logging.info(
        f"Minority class counts: Train+Val={n_trainval_minority} buggy, "
        f"Holdout={n_holdout_minority} buggy"
    )

    # Check if hold-out has both classes
    if len(y_holdout.unique()) < 2:
        logging.warning(f"Hold-out test set has only one class. Final test metrics may be limited.")

    # --- Adaptive Fold Count ---
    n_train_val_commits = len(train_val_commits)
    n_train_val_positive = int(y_train_val.sum())
    outer_folds, inner_folds = get_adaptive_fold_counts(
        len(X_train_val), n_train_val_commits, n_train_val_positive
    )

    # Override with user-specified folds if provided (but warn if potentially problematic)
    if n_folds != 5:  # User specified custom folds
        if n_folds > outer_folds:
            logging.warning(
                f"User-specified {n_folds} folds may be too many for {n_train_val_commits} commits. "
                f"Using adaptive recommendation: {outer_folds} folds."
            )
        else:
            outer_folds = n_folds
            logging.info(f"Using user-specified {outer_folds} outer folds.")

    # Get min_class_ratio from CLI args
    min_class_ratio = cli_args.min_class_ratio if cli_args and hasattr(cli_args, 'min_class_ratio') else 0.05

    # Check if shuffle CV is requested
    use_shuffle_cv = cli_args.shuffle_cv if cli_args and hasattr(cli_args, 'shuffle_cv') else False

    if use_shuffle_cv:
        # Use stratified shuffle CV instead of temporal CV
        from sklearn.model_selection import StratifiedKFold
        outer_cv = StratifiedKFold(n_splits=outer_folds, shuffle=True, random_state=42)
        logging.info(f"Using Stratified Shuffle CV with {outer_folds} folds (temporal order ignored)")
        cv_type = 'shuffle'
    else:
        # Create temporal CV splitter for outer loop
        outer_cv = CommitGroupTimeSeriesSplit(
            n_splits=outer_folds,
            min_train_ratio=0.15,
            gap=0,
            min_class_ratio=min_class_ratio
        )
        logging.info(f"Using Temporal CV with {outer_folds} folds")
        cv_type = 'temporal'

    if min_class_ratio > 0 and not use_shuffle_cv:
        logging.info(f"Minimum class ratio threshold: {min_class_ratio:.2%} (folds below this will be skipped)")

    # Ensure the 'resampling' key in the results dict uses the path name ('none' or method name)
    results = {
        'project': project_name,
        'level': level,
        'resampling': resampling_strategy if resampling_strategy is not None else 'none',
        'cv_type': cv_type,
        'n_unique_commits': n_unique_commits,
        'n_train_val_commits': n_train_val_commits,
        'n_holdout_commits': len(holdout_commits),
        'outer_folds': outer_folds,
        'inner_folds': inner_folds,
        'min_class_ratio': min_class_ratio if not use_shuffle_cv else None
    }

    # --- Feature Selection Configuration ---
    # Feature selection will be applied INSIDE each outer fold to prevent leakage
    # This means feature selection is fit on outer-train and applied to outer-test
    fs_method = cli_args.select_feature if cli_args and cli_args.select_feature else None
    k_fs = cli_args.k_features if cli_args and cli_args.k_features else None

    if fs_method:
        logging.info(
            f"Dynamic feature selection ENABLED: method='{fs_method}', k={k_fs}. "
            f"Feature selection will be applied within each outer fold (fit on train, transform on test)."
        )

    # NOTE: Feature selection will be applied INSIDE each outer fold to prevent data leakage.
    # This ensures outer-test fold data never influences feature selection.

    # If after feature selection, X_train_val is empty, skip further processing
    if X_train_val.empty:
        logging.warning(f"No features remaining after dynamic selection for project {project_name}, level {level}, resampling {resampling_strategy}. Skipping model training for this configuration.")
        results['error'] = "No features remaining after dynamic selection"
        return results

    all_classifier_funcs = {
        'naive_bayes': analyze_with_naive_bayes, 'xgboost': analyze_with_xgboost,
        'random_forest': analyze_with_random_forest,
        'logistic_regression': analyze_with_logistic_regression, 'catboost': analyze_with_catboost,
        'lightgbm': analyze_with_lightgbm, 'gradient_boosting': analyze_with_gradient_boosting,
        'decision_tree': analyze_with_decision_tree, 'voting': analyze_with_voting,
        'mlp': analyze_with_mlp, 'stacking': analyze_with_stacking
    }

    # Filter classifiers if methods_to_run is specified
    if methods_to_run == 'all' or methods_to_run is None:
        classifier_funcs_to_run = all_classifier_funcs
    else:
        if isinstance(methods_to_run, str):
            methods_to_run = [methods_to_run]
        classifier_funcs_to_run = {
            name: func for name, func in all_classifier_funcs.items()
            if name in methods_to_run
        }
    if not classifier_funcs_to_run:
        logging.warning("No valid ML methods specified to run. Skipping analysis.")
        return None

    roc_data = {name: [] for name in classifier_funcs_to_run.keys()}
    plot_feature_correlations(X_train_val, output_dir, project_name, level)

    # --- Save temporal validation info ---
    temporal_validation_info = {
        'project': project_name,
        'level': level,
        'n_total_samples': len(X),
        'n_unique_commits': n_unique_commits,
        'n_train_val_commits': n_train_val_commits,
        'n_holdout_commits': len(holdout_commits),
        'n_train_val_samples': len(X_train_val),
        'n_holdout_samples': len(X_holdout),
        'train_val_max_timestamp': int(train_val_max_ts),
        'holdout_min_timestamp': int(holdout_min_ts),
        'temporal_gap_seconds': int(holdout_min_ts - train_val_max_ts),
        'holdout_temporal_valid': bool(train_val_max_ts < holdout_min_ts),
        'outer_folds': outer_folds,
        'inner_folds': inner_folds,
        'cv_type': cv_type,
        'fold_validations': []
    }

    # --- CV Fold Preparation with Validation ---
    process_args = []
    fold_count = 0

    try:
        # Different split call based on CV type
        if use_shuffle_cv:
            fold_iterator = outer_cv.split(X_train_val, y_train_val)
        else:
            fold_iterator = outer_cv.split(
                X_train_val, y_train_val, groups=train_val_groups, timestamps=train_val_timestamps
            )

        for fold, (train_idx, test_idx) in enumerate(fold_iterator):
            fold_count += 1

            # Get class distribution
            class_dist = get_fold_class_distribution(y_train_val, train_idx, test_idx, fold_num=fold+1)

            if use_shuffle_cv:
                # For shuffle CV, skip temporal and group validation
                fold_validation = {
                    'fold': fold + 1,
                    'temporal': {'is_valid': True, 'note': 'Shuffle CV - temporal validation skipped'},
                    'group_integrity': {'is_valid': True, 'note': 'Shuffle CV - group validation skipped'},
                    'class_distribution': class_dist
                }
                temporal_validation_info['fold_validations'].append(fold_validation)

                logging.info(
                    f"[Fold {fold+1}] Train: {len(train_idx)} samples ({class_dist['train_positive']} pos, "
                    f"{class_dist['train_positive_ratio']:.1%} bug), "
                    f"Test: {len(test_idx)} samples ({class_dist['test_positive']} pos, "
                    f"{class_dist['test_positive_ratio']:.1%} bug)"
                )
            else:
                # Validate temporal integrity
                temporal_result = outer_cv.validate_temporal_integrity(
                    train_val_timestamps, train_idx, test_idx, fold_num=fold+1
                )

                # Validate commit group integrity
                group_result = validate_commit_group_integrity(
                    train_val_groups, train_idx, test_idx, fold_num=fold+1
                )

                fold_validation = {
                    'fold': fold + 1,
                    'temporal': temporal_result,
                    'group_integrity': group_result,
                    'class_distribution': class_dist
                }
                temporal_validation_info['fold_validations'].append(fold_validation)

                # Log fold info
                logging.info(
                    f"[Fold {fold+1}] Train: {len(train_idx)} samples ({class_dist['train_positive']} pos), "
                    f"Test: {len(test_idx)} samples ({class_dist['test_positive']} pos), "
                    f"Temporal valid: {temporal_result['is_valid']}, Group valid: {group_result['is_valid']}"
                )

                if not temporal_result['is_valid'] or not group_result['is_valid']:
                    logging.error(f"[Fold {fold+1}] Validation failed! Skipping this fold.")
                    continue

            X_train_fold = X_train_val.iloc[train_idx].reset_index(drop=True)
            X_test_fold = X_train_val.iloc[test_idx].reset_index(drop=True)
            y_train_fold = y_train_val.iloc[train_idx].reset_index(drop=True)
            y_test_fold = y_train_val.iloc[test_idx].reset_index(drop=True)

            # --- Apply Feature Selection INSIDE the fold (fit on train, transform on test) ---
            fold_fs_metadata = None
            if fs_method:
                try:
                    X_train_fold, X_test_fold, selected_features, fold_fs_metadata = apply_feature_selection_on_fold(
                        X_train_fold, y_train_fold, X_test_fold,
                        fs_method=fs_method,
                        k_features=k_fs,
                        output_dir=output_dir,
                        fold_num=fold+1
                    )
                    logging.info(
                        f"[Fold {fold+1}] Feature selection applied: {len(selected_features)} features selected "
                        f"using method '{fs_method}'"
                    )
                except Exception as e_fs:
                    logging.error(f"[Fold {fold+1}] Feature selection failed: {e_fs}. Using all features.")

            # Apply resampling to this fold's training data
            X_train_resampled, y_train_resampled = X_train_fold.copy(), y_train_fold.copy()
            if resampling_strategy and resampling_strategy != 'none':
                logging.debug(f"[Fold {fold+1}] Applying resampling: {resampling_strategy}")
                try:
                    resampler = get_resampling_method(resampling_strategy)
                    if resampler:
                        X_train_resampled, y_train_resampled = resampler.fit_resample(X_train_resampled, y_train_resampled)
                        print_class_distribution(f"Fold {fold+1} - After Resampling ({resampling_strategy})", y_train_resampled)
                except Exception as e:
                    logging.error(f"Error during resampling with {resampling_strategy} in fold {fold+1}: {e}")

            # Create tasks for all models for THIS fold
            for model_name, clf_func in classifier_funcs_to_run.items():
                process_args.append((
                    model_name, clf_func, X_train_resampled.copy(), X_test_fold.copy(),
                    y_train_resampled.copy(), y_test_fold.copy(),
                    project_name, level, resampling_strategy, optimize, output_dir, fold
            ))
    except ValueError as e:
        logging.error(f"Error during temporal CV split: {e}")
        temporal_validation_info['error'] = str(e)

    # Save temporal validation info
    temporal_validation_path = output_dir / "temporal_validation_info.json"
    try:
        with open(temporal_validation_path, 'w') as f:
            json.dump(convert_numpy_to_list_recursive(temporal_validation_info), f, indent=2)
        logging.info(f"Temporal validation info saved to {temporal_validation_path}")
    except Exception as e:
        logging.error(f"Error saving temporal validation info: {e}")

    if not process_args:
        logging.error(f"No valid folds created for {project_name}. Cannot proceed with analysis.")
        results['error'] = "No valid temporal CV folds could be created"
        return results

    # --- Run all tasks in parallel ---
    num_processes = min(os.cpu_count() if os.cpu_count() else 1, 6)
    logging.info(f"Running {len(process_args)} fold-analyses with up to {num_processes} parallel processes.")

    fold_results_list = []

    folds_task_id = None
    if progress:
        folds_task_id = progress.add_task(f"[blue]  Folds for {project_name}...", total=len(process_args))

    with multiprocessing.Pool(processes=num_processes) as pool:
        results_iterator = pool.imap_unordered(single_fold_wrapper_for_imap, process_args)
        for res in results_iterator:
            fold_results_list.append(res)
            if progress and folds_task_id is not None:
                progress.update(folds_task_id, advance=1)

    if progress and folds_task_id is not None:
        progress.remove_task(folds_task_id)

    # --- Aggregate results ---
    model_fold_results = {model_name: [] for model_name in classifier_funcs_to_run.keys()}
    for result_tuple in fold_results_list:
        if result_tuple:
            model_name, _, metrics = result_tuple
            model_fold_results[model_name].append(metrics)

    # --- Hold-out test evaluation storage ---
    holdout_results = {}
    best_models_for_holdout = {}  # Store best model from CV for hold-out evaluation

    # Process aggregated results for each model
    for model_name, fold_metrics_list in model_fold_results.items():
        if not fold_metrics_list:
            logging.error(f"No metrics collected for any fold for model {model_name}. Skipping aggregation.")
            results[model_name] = {'error': 'No fold metrics collected'}
            continue

        # --- Optimization Results Saving ---
        if optimize and fold_metrics_list and any('best_cv_score' in d for d in fold_metrics_list):
            logging.info(f"Aggregating and saving optimization results for model: {model_name}")

            all_results_to_log = []
            for fold_data in fold_metrics_list:
                if 'best_cv_score' not in fold_data or fold_data.get('best_params') is None:
                    continue

                test_perf = {
                    'accuracy': fold_data.get('accuracy'),
                    'f1_score': fold_data.get('f1_1'),
                    'precision': fold_data.get('precision_1'),
                    'recall': fold_data.get('recall_1'),
                    'auc': fold_data.get('auc'),
                    'f1_0': fold_data.get('f1_0'),
                    'precision_0': fold_data.get('precision_0'),
                    'recall_0': fold_data.get('recall_0')
                }
                res = {
                    'project': project_name,
                    'level': level,
                    'resampling_method': resampling_strategy if resampling_strategy is not None else 'none',
                    'model': model_name,
                    'best_parameters': fold_data.get('best_params'),
                    'best_cv_score': fold_data.get('best_cv_score'),
                    'timestamp': datetime.datetime.now().isoformat(),
                    'test_performance': {k: v for k, v in test_perf.items() if v is not None}
                }
                all_results_to_log.append(res)

            if all_results_to_log:
                optimization_dir = output_dir / 'optimization_results'
                optimization_dir.mkdir(parents=True, exist_ok=True)
                resampling_name = resampling_strategy if resampling_strategy is not None else 'none'

                combined_file_path = optimization_dir / f"{project_name}_{level}_{resampling_name}_all_optimizations.json"
                try:
                    with open(combined_file_path, 'w', encoding='utf-8') as f:
                        json.dump(convert_numpy_to_list_recursive(all_results_to_log), f, indent=2, ensure_ascii=False)
                    logging.info(f"All fold optimization results for '{model_name}' saved to {combined_file_path}")
                except Exception as e:
                    logging.error(f"Error saving all fold optimization results to {combined_file_path}: {e}")

                best_fold_result_data = max(all_results_to_log, key=lambda x: x.get('test_performance', {}).get('f1_score', 0))
                single_file_path = optimization_dir / f"{project_name}_{level}_{resampling_name}_{model_name}_optimization.json"
                try:
                    with open(single_file_path, 'w', encoding='utf-8') as f:
                        json.dump(convert_numpy_to_list_recursive(best_fold_result_data), f, indent=2, ensure_ascii=False)
                    logging.info(f"Best optimization result for '{model_name}' saved to {single_file_path}")
                except Exception as e:
                    logging.error(f"Error saving single best optimization result to {single_file_path}: {e}")

        # --- Compute Mean AND Standard Deviation for CV metrics ---
        metrics_to_average_list = []
        for metrics_dict in fold_metrics_list:
            if 'error' not in metrics_dict:
                avg_candidate_metrics = {
                    k: v for k, v in metrics_dict.items()
                    if k not in ['feature_importance', 'fpr', 'tpr', 'fold', 'error', 'y_test_fold', 'y_prob_fold', 'best_params', 'best_cv_score']
                }
                numeric_metrics_for_avg = {k: v for k, v in avg_candidate_metrics.items() if isinstance(v, (int, float, np.number))}
                if numeric_metrics_for_avg:
                    metrics_to_average_list.append(numeric_metrics_for_avg)

        if not metrics_to_average_list:
            logging.error(f"No valid metrics to average for model {model_name}.")
            avg_metrics = {'error': "No valid fold metrics to average"}
            std_metrics = {}
        else:
            avg_metrics_df = pd.DataFrame(metrics_to_average_list)
            numeric_cols = avg_metrics_df.select_dtypes(include=np.number).columns

            # Calculate mean
            avg_metrics = avg_metrics_df[numeric_cols].mean().to_dict()

            # Calculate standard deviation
            std_metrics = avg_metrics_df[numeric_cols].std().to_dict()

            # Add std metrics with _std suffix
            for key, value in std_metrics.items():
                avg_metrics[f'{key}_std'] = value

            # Add number of folds used
            avg_metrics['n_folds_used'] = len(metrics_to_average_list)

            logging.info(
                f"[{model_name}] CV Results - "
                f"F1: {avg_metrics.get('f1_1', 0):.4f} ± {std_metrics.get('f1_1', 0):.4f}, "
                f"Accuracy: {avg_metrics.get('accuracy', 0):.4f} ± {std_metrics.get('accuracy', 0):.4f}, "
                f"ROC-AUC: {avg_metrics.get('auc', 0):.4f} ± {std_metrics.get('auc', 0):.4f}, "
                f"PR-AUC: {avg_metrics.get('pr_auc', 0):.4f} ± {std_metrics.get('pr_auc', 0):.4f}, "
                f"MCC: {avg_metrics.get('mcc', 0):.4f} ± {std_metrics.get('mcc', 0):.4f}"
            )

        # --- Average Feature Importances ---
        fold_importances_list = [m['feature_importance'] for m in fold_metrics_list if m and 'feature_importance' in m and m['feature_importance'] is not None]
        if fold_importances_list:
            try:
                if all(isinstance(fi, (list, np.ndarray)) and len(fi) == len(fold_importances_list[0]) for fi in fold_importances_list):
                    avg_metrics['feature_importance'] = np.mean([np.array(fi) for fi in fold_importances_list], axis=0).tolist()
            except Exception as e:
                logging.warning(f"Could not average feature importances for {model_name}: {e}")

        results[model_name] = avg_metrics

        # --- Collect ROC Data ---
        model_roc_data = []
        for fold_metrics in fold_metrics_list:
            if fold_metrics and 'y_test_fold' in fold_metrics and 'y_prob_fold' in fold_metrics:
                model_roc_data.append({
                    'y_test': fold_metrics['y_test_fold'],
                    'y_prob': fold_metrics['y_prob_fold'],
                    'auc': fold_metrics.get('auc', 0.0),
                    'fold': fold_metrics.get('fold', 'N/A')
                })
        roc_data[model_name] = model_roc_data

        # --- Save Detailed Fold Metrics to JSON ---
        fold_metrics_file_path = output_dir / f"{model_name}_fold_metrics.json"
        try:
            serializable_fold_metrics = convert_numpy_to_list_recursive(fold_metrics_list)
            with open(fold_metrics_file_path, 'w') as f:
                json.dump(serializable_fold_metrics, f, indent=4)
            logging.info(f"Successfully saved fold metrics for {model_name} to {fold_metrics_file_path}")
        except Exception as e:
            logging.error(f"CRITICAL ERROR saving fold metrics for {model_name} to {fold_metrics_file_path}: {e}", exc_info=True)

    # --- Hold-out Test Evaluation ---
    logging.info("=" * 60)
    logging.info("Starting Hold-out Test Set Evaluation (Final 20%)")
    logging.info("=" * 60)

    # Prepare copies for hold-out evaluation
    X_train_val_for_holdout = X_train_val.copy()
    X_holdout_for_eval = X_holdout.copy()

    # --- Apply Feature Selection for Hold-out (fit on full train_val, transform on holdout) ---
    holdout_fs_metadata = None
    if fs_method:
        try:
            X_train_val_for_holdout, X_holdout_for_eval, selected_features_holdout, holdout_fs_metadata = apply_feature_selection_on_fold(
                X_train_val_for_holdout, y_train_val, X_holdout_for_eval,
                fs_method=fs_method,
                k_features=k_fs,
                output_dir=output_dir,
                fold_num='holdout'
            )
            logging.info(
                f"[Holdout] Feature selection applied: {len(selected_features_holdout)} features selected "
                f"using method '{fs_method}'"
            )
        except Exception as e_fs:
            logging.error(f"[Holdout] Feature selection failed: {e_fs}. Using all features.")

    # Train final models on full train_val set and evaluate on hold-out
    for model_name, clf_func in classifier_funcs_to_run.items():
        logging.info(f"Training final {model_name} model on full train/val set for hold-out evaluation...")

        try:
            # Apply resampling to full train_val set
            X_train_final, y_train_final = X_train_val_for_holdout.copy(), y_train_val.copy()
            if resampling_strategy and resampling_strategy != 'none':
                try:
                    resampler = get_resampling_method(resampling_strategy)
                    if resampler:
                        X_train_final, y_train_final = resampler.fit_resample(X_train_final, y_train_final)
                        logging.info(f"Applied {resampling_strategy} to full train/val set for {model_name}")
                except Exception as e:
                    logging.error(f"Error applying resampling for hold-out evaluation: {e}")

            # Train and evaluate on hold-out
            holdout_metrics_with_model = clf_func(
                X_train_final, X_holdout_for_eval, y_train_final, y_holdout,
                project_name, output_dir, optimize=optimize, level=level,
                resampling_method=resampling_strategy
            )

            if holdout_metrics_with_model and holdout_metrics_with_model.metrics:
                holdout_model_metrics = holdout_metrics_with_model.metrics.copy()
                # Remove fold-specific keys
                holdout_model_metrics.pop('y_test_fold', None)
                holdout_model_metrics.pop('y_prob_fold', None)
                holdout_model_metrics.pop('fold', None)
                holdout_model_metrics['evaluation_type'] = 'holdout_test'

                # Compute bootstrap CI for holdout metrics if requested
                holdout_bootstrap_ci = None
                if compute_bootstrap_ci and len(np.unique(y_holdout)) > 1:
                    try:
                        # Get predictions from the trained model
                        y_prob = holdout_model_metrics.get('y_prob_fold', None)
                        if y_prob is None and hasattr(holdout_metrics_with_model, 'model'):
                            # Try to get predictions from model
                            model = holdout_metrics_with_model.model
                            if hasattr(model, 'predict_proba'):
                                y_prob = model.predict_proba(X_holdout_for_eval)[:, 1]

                        if y_prob is not None:
                            y_pred = (np.array(y_prob) >= 0.5).astype(int)
                            holdout_bootstrap_ci = compute_all_bootstrap_cis(
                                np.array(y_holdout), y_pred, np.array(y_prob),
                                n_bootstrap=n_bootstrap,
                                confidence_level=0.95,
                                random_state=42
                            )
                            holdout_model_metrics['bootstrap_ci'] = holdout_bootstrap_ci
                            logging.info(f"[{model_name}] Bootstrap CI computed for holdout metrics")
                    except Exception as e:
                        logging.warning(f"[{model_name}] Could not compute bootstrap CI: {e}")

                holdout_results[model_name] = holdout_model_metrics

                # Add hold-out results to main results dict with prefix
                for key, value in holdout_model_metrics.items():
                    if key not in ['feature_importance', 'evaluation_type', 'best_params', 'best_cv_score']:
                        results[model_name][f'holdout_{key}'] = value

                logging.info(
                    f"[{model_name}] Hold-out Test Results - "
                    f"F1: {holdout_model_metrics.get('f1_1', 0):.4f}, "
                    f"Accuracy: {holdout_model_metrics.get('accuracy', 0):.4f}, "
                    f"AUC: {holdout_model_metrics.get('auc', 0):.4f}"
                )
            else:
                logging.warning(f"No hold-out metrics returned for {model_name}")
                holdout_results[model_name] = {'error': 'No hold-out metrics returned'}

        except Exception as e:
            logging.error(f"Error during hold-out evaluation for {model_name}: {e}", exc_info=True)
            holdout_results[model_name] = {'error': str(e)}

    # Save hold-out results
    holdout_results_path = output_dir / "holdout_test_results.json"
    try:
        with open(holdout_results_path, 'w') as f:
            json.dump(convert_numpy_to_list_recursive(holdout_results), f, indent=4)
        logging.info(f"Hold-out test results saved to {holdout_results_path}")
    except Exception as e:
        logging.error(f"Error saving hold-out results: {e}")

    # Save ROC fold data to JSON for potential regeneration
    roc_data_path = output_dir / "roc_fold_data.json"
    try:
        serializable_roc_data = convert_numpy_to_list_recursive(roc_data)
        with open(roc_data_path, 'w') as f:
            json.dump(serializable_roc_data, f, indent=4)
        logging.info(f"ROC fold data saved to {roc_data_path}")
    except Exception as e:
        logging.error(f"Error saving ROC fold data to {roc_data_path}: {e}")

    plot_roc_curves(roc_data, project_name, level, output_dir)

    # Plot F1 scores and Accuracy
    plot_scores_barchart(results, 'f1_1', 'F1 Scores (Class 1) - CV Mean', f'{project_name}_{level}_f1_scores', project_name, level, output_dir)
    plot_scores_barchart(results, 'accuracy', 'Accuracy Scores - CV Mean', f'{project_name}_{level}_accuracy_scores', project_name, level, output_dir)

    # Add summary to results
    results['temporal_validation'] = {
        'holdout_temporal_valid': temporal_validation_info.get('holdout_temporal_valid', False),
        'n_valid_folds': len([f for f in temporal_validation_info.get('fold_validations', [])
                             if f.get('temporal', {}).get('is_valid', False) and
                                f.get('group_integrity', {}).get('is_valid', False)])
    }

    # --- Save Analysis Summary JSON (for academic paper) ---
    analysis_summary = {
        'project': project_name,
        'level': level,
        'resampling_strategy': resampling_strategy if resampling_strategy else 'none',
        'cv_type': cv_type,
        'n_outer_folds': outer_folds,
        'n_inner_folds': inner_folds,
        'dataset_info': {
            'n_total_samples': len(X),
            'n_train_val_samples': len(X_train_val),
            'n_holdout_samples': len(X_holdout),
            'train_val_bug_count': int(y_train_val.sum()),
            'train_val_bug_ratio': float(y_train_val.mean()),
            'holdout_bug_count': int(y_holdout.sum()),
            'holdout_bug_ratio': float(y_holdout.mean()),
            'n_features': X_train_val.shape[1],
            'feature_names': list(X_train_val.columns)
        },
        # Dataset quality assessment for statistical analysis decisions
        'dataset_quality': {
            'quality_level': dataset_quality,  # 'primary', 'exploratory', or 'excluded'
            'trainval_minority_count': n_trainval_minority,
            'holdout_minority_count': n_holdout_minority,
            'is_primary': dataset_quality == DatasetQuality.PRIMARY,
            'is_exploratory': dataset_quality == DatasetQuality.EXPLORATORY,
            'issues': quality_reasons.get('issues', []),
            'thresholds_used': {
                'min_trainval_minority_primary': MIN_TRAINVAL_MINORITY_PRIMARY,
                'min_holdout_minority_primary': MIN_HOLDOUT_MINORITY_PRIMARY,
                'min_trainval_minority_exploratory': MIN_TRAINVAL_MINORITY_EXPLORATORY,
                'min_holdout_minority_exploratory': MIN_HOLDOUT_MINORITY_EXPLORATORY
            }
        },
        'fold_class_distributions': [],
        'models': {}
    }

    # Extract fold class distributions from temporal_validation_info
    for fold_info in temporal_validation_info.get('fold_validations', []):
        class_dist = fold_info.get('class_distribution', {})
        analysis_summary['fold_class_distributions'].append({
            'fold': fold_info.get('fold'),
            'train_samples': class_dist.get('train_total', 0),
            'train_positive': class_dist.get('train_positive', 0),
            'train_positive_ratio': class_dist.get('train_positive_ratio', 0),
            'test_samples': class_dist.get('test_total', 0),
            'test_positive': class_dist.get('test_positive', 0),
            'test_positive_ratio': class_dist.get('test_positive_ratio', 0)
        })

    # Add feature selection metadata if available
    if holdout_fs_metadata:
        analysis_summary['feature_selection'] = holdout_fs_metadata
    elif fs_method:
        # Feature selection was requested but no metadata available
        analysis_summary['feature_selection'] = {
            'method': fs_method,
            'k_requested': k_fs if k_fs else 'auto',
            'note': 'Feature selection was applied but detailed metadata not captured'
        }
    else:
        analysis_summary['feature_selection'] = None

    # Extract model metrics for summary
    for model_name, model_metrics in results.items():
        if model_name in ['temporal_validation', 'project', 'level', 'resampling', 'cv_type', 'error']:
            continue
        if isinstance(model_metrics, dict) and 'error' not in model_metrics:
            model_summary = {
                'cv_metrics': {
                    'accuracy': model_metrics.get('accuracy'),
                    'accuracy_std': model_metrics.get('accuracy_std'),
                    'f1_bug': model_metrics.get('f1_1'),
                    'f1_bug_std': model_metrics.get('f1_1_std'),
                    'precision_bug': model_metrics.get('precision_1'),
                    'precision_bug_std': model_metrics.get('precision_1_std'),
                    'recall_bug': model_metrics.get('recall_1'),
                    'recall_bug_std': model_metrics.get('recall_1_std'),
                    'roc_auc': model_metrics.get('auc'),
                    'roc_auc_std': model_metrics.get('auc_std'),
                    'pr_auc': model_metrics.get('pr_auc'),
                    'pr_auc_std': model_metrics.get('pr_auc_std'),
                    'mcc': model_metrics.get('mcc'),
                    'mcc_std': model_metrics.get('mcc_std')
                },
                'holdout_metrics': {
                    'accuracy': model_metrics.get('holdout_accuracy'),
                    'f1_bug': model_metrics.get('holdout_f1_1'),
                    'precision_bug': model_metrics.get('holdout_precision_1'),
                    'recall_bug': model_metrics.get('holdout_recall_1'),
                    'roc_auc': model_metrics.get('holdout_auc'),
                    'pr_auc': model_metrics.get('holdout_pr_auc'),
                    'mcc': model_metrics.get('holdout_mcc')
                }
            }

            # Add bootstrap CI if available
            holdout_ci = holdout_results.get(model_name, {}).get('bootstrap_ci')
            if holdout_ci:
                model_summary['holdout_bootstrap_ci'] = holdout_ci

            analysis_summary['models'][model_name] = model_summary

    # Save analysis summary
    analysis_summary_path = output_dir / "analysis_summary.json"
    try:
        with open(analysis_summary_path, 'w') as f:
            json.dump(convert_numpy_to_list_recursive(analysis_summary), f, indent=2)
        logging.info(f"Analysis summary saved to {analysis_summary_path}")
    except Exception as e:
        logging.error(f"Error saving analysis summary: {e}")

    return results

def single_fold_wrapper_for_imap(args):
    """Unpacks arguments and calls the real worker. For use with imap_unordered."""
    return run_single_fold_analysis_wrapper(*args)

# Wrapper function for multiprocessing
def run_single_fold_analysis_wrapper(model_name, clf_func, X_train_fold, X_test_fold, y_train_fold, y_test_fold, project_name, level, resampling_strategy, optimize, output_dir, fold_num):
    """Wrapper to run analysis for a single model on a single fold."""
    # This is the target for multiprocessing.Pool.apply_async
    logging.info(f"[Process {os.getpid()}] Starting evaluation for {model_name}, Fold {fold_num+1} on {project_name} ({level})...")

    # Delegate to the existing classifier-specific analysis function (e.g., analyze_with_naive_bayes)
    # which in turn calls _run_analysis. This keeps the logic for optimization, scaling, etc., encapsulated.
    metrics_with_model = clf_func(
        X_train_fold, X_test_fold, y_train_fold, y_test_fold,
        project_name, output_dir, optimize=optimize, level=level,
        resampling_method=resampling_strategy
    )

    if metrics_with_model and metrics_with_model.metrics:
        current_fold_metrics = metrics_with_model.metrics.copy()
        current_fold_metrics['fold'] = fold_num + 1 # Use 1-based fold number for consistency

        # Add train class distribution (after resampling if applied)
        n_train_total = len(y_train_fold)
        n_train_positive = int(np.sum(y_train_fold))
        current_fold_metrics['train_class_distribution'] = {
            'n_total': n_train_total,
            'n_positive': n_train_positive,
            'n_negative': n_train_total - n_train_positive,
            'positive_ratio': n_train_positive / n_train_total if n_train_total > 0 else 0
        }

        return model_name, fold_num, current_fold_metrics
    else:
        logging.warning(f"No metrics returned for {model_name} in fold {fold_num+1}.")
        return model_name, fold_num, {'error': f'No metrics for fold {fold_num+1}', 'fold': fold_num+1}

def print_class_distribution(title, y):
    if y is None or len(y) == 0:
        logging.warning(f"{title}: Empty target series.")
        return
    total_samples = len(y)
    bug_samples = y.sum()
    non_bug_samples = total_samples - bug_samples
    logging.info(f"{title} - Total: {total_samples}, Bugs: {bug_samples} ({(bug_samples/total_samples)*100:.2f}%), Non-Bugs: {non_bug_samples} ({(non_bug_samples/total_samples)*100:.2f}%)")
    if bug_samples > 0 and non_bug_samples > 0 :
         logging.info(f"Class imbalance ratio (Non-Bug/Bug): {non_bug_samples/bug_samples:.2f}:1")


def collect_dataset_statistics(project_name, level):
    df = load_project_data(project_name, level)
    if df is None: return None
    total_samples = len(df)
    bug_samples = df['is_bug'].sum()
    non_bug_samples = total_samples - bug_samples
    return {
        'project': project_name, 'level': level, 'total_samples': total_samples,
        'bug_samples': int(bug_samples), 'non_bug_samples': int(non_bug_samples),
        'bug_percentage': (bug_samples/total_samples)*100 if total_samples > 0 else 0,
        'class_imbalance_ratio': non_bug_samples/bug_samples if bug_samples > 0 else float('inf')
    }

def format_metric(value):
    """Format metric values for markdown tables."""
    if pd.isna(value) or value is None:
        return "N/A"
    if isinstance(value, (int, float)):
        return f"{value:.4f}"
    return str(value)


def collect_results_from_hierarchical_structure(level, cv_type, feature_set, project_filter=None):
    """
    Collect results from hierarchical directory structure.

    Structure: results_{level}_level/{project}/{cv_type}/{feature_set}/{resampling}/

    Returns list of DataFrames with results.
    """
    results_dir = get_results_dir(level)
    all_data = []

    # Pattern: {project}/{cv_type}/{feature_set}/{resampling}
    for project_dir in results_dir.iterdir():
        if not project_dir.is_dir() or project_dir.name.startswith('_'):
            continue

        if project_filter and project_filter != 'all' and project_dir.name != project_filter:
            continue

        project_name = project_dir.name
        cv_dir = project_dir / cv_type

        if not cv_dir.exists():
            continue

        feature_dir = cv_dir / feature_set
        if not feature_dir.exists():
            continue

        # Iterate over resampling strategies
        for resampling_dir in feature_dir.iterdir():
            if not resampling_dir.is_dir():
                continue

            resampling_name = resampling_dir.name

            # Collect metrics from fold_metrics.json files
            for model_name in ALL_CLASSIFIER_FUNCTION_NAMES:
                fold_metrics_file = resampling_dir / f"{model_name}_fold_metrics.json"

                if not fold_metrics_file.exists():
                    continue

                try:
                    with open(fold_metrics_file, 'r') as f:
                        fold_metrics = json.load(f)

                    # Calculate mean metrics across folds
                    if fold_metrics:
                        metrics_to_avg = ['accuracy', 'precision_1', 'recall_1', 'f1_1',
                                         'precision_0', 'recall_0', 'f1_0', 'auc', 'mcc', 'pr_auc']

                        row_data = {
                            'project': project_name,
                            'level': level,
                            'cv_type': cv_type,
                            'feature_set': feature_set,
                            'resampling': resampling_name,
                            'model': model_name
                        }

                        for metric in metrics_to_avg:
                            values = [fm.get(metric) for fm in fold_metrics if fm.get(metric) is not None]
                            if values:
                                row_data[f'{model_name}_{metric}'] = np.mean(values)
                                row_data[f'{model_name}_{metric}_std'] = np.std(values)

                        all_data.append(pd.DataFrame([row_data]))

                except Exception as e:
                    logging.error(f"Error reading {fold_metrics_file}: {e}")

    return all_data


def generate_markdown_tables(levels_to_process, project_name_filter=None, cli_args=None):
    """
    Generate markdown tables from existing analysis results.

    Uses new hierarchical structure:
        results_{level}_level/{project}/{cv_type}/{feature_set}/{resampling}/
    """
    logging.info(f"Generating markdown tables for levels: {levels_to_process}")

    # Determine CV type and feature set from cli_args
    cv_type = get_cv_type_name(cli_args and getattr(cli_args, 'shuffle_cv', False))
    feature_set = get_feature_set_name(cli_args and getattr(cli_args, 'exclude_go_metrics', False))

    for level in levels_to_process:
        logging.info(f"Processing level: {level}")
        results_dir = get_results_dir(level)

        # New summary structure: _summary/{cv_type}/{feature_set}/
        summary_dir = results_dir / "_summary" / cv_type / feature_set

        if not summary_dir.exists():
            # Try to collect from hierarchical analysis directories
            logging.info(f"Summary directory not found, collecting from analysis directories...")
            all_data = collect_results_from_hierarchical_structure(
                level, cv_type, feature_set, project_name_filter
            )
        else:
            # Collect all CSV files from resampling subdirectories
            all_data = []

            for resampling_dir in summary_dir.iterdir():
                if not resampling_dir.is_dir():
                    continue

                for csv_file in resampling_dir.glob("classification_summary_*.csv"):
                    try:
                        df = pd.read_csv(csv_file)
                        if not df.empty:
                            all_data.append(df)
                            logging.info(f"Loaded {len(df)} rows from {csv_file}")
                    except Exception as e:
                        logging.error(f"Error reading {csv_file}: {e}")

        if not all_data:
            logging.warning(f"No data found for level {level}")
            continue

        # Combine all data
        combined_df = pd.concat(all_data, ignore_index=True)

        # Exclude 'combined' and 'combine' projects from tables
        combined_df = combined_df[~combined_df['project'].isin(['combined', 'combine'])]
        if combined_df.empty:
            logging.warning(f"No data remaining after filtering out 'combined'/'combine' projects for level {level}")
            continue

        # Filter by project if specified
        if project_name_filter and project_name_filter != 'all':
            combined_df = combined_df[combined_df['project'] == project_name_filter]
            if combined_df.empty:
                logging.warning(f"No data found for project {project_name_filter} at level {level}")
                continue

        # Melt the dataframe to long format for easier processing
        id_vars = ['project', 'level', 'resampling']
        value_vars = [col for col in combined_df.columns if col not in id_vars]

        # Split model_metric columns
        melted_rows = []
        for _, row in combined_df.iterrows():
            base_info = {col: row[col] for col in id_vars}

            # Group metrics by model
            models = {}
            for col in value_vars:
                if '_' in col:
                    # Find the correct model name by checking against known classifier names
                    model_name = None
                    for classifier_name in ALL_CLASSIFIER_FUNCTION_NAMES:
                        if col.startswith(classifier_name + '_'):
                            model_name = classifier_name
                            metric_name = col[len(classifier_name) + 1:]  # Remove model_name + '_'
                            break

                    if model_name:
                        if model_name not in models:
                            models[model_name] = {}
                        models[model_name][metric_name] = row[col]

            # Create a row for each model
            for model_name, metrics in models.items():
                row_data = base_info.copy()
                row_data['model'] = model_name
                row_data.update(metrics)
                melted_rows.append(row_data)

        if not melted_rows:
            logging.warning(f"No melted data available for level {level}")
            continue

        melted_df = pd.DataFrame(melted_rows)

        # Generate top-level results.md
        generate_top_level_results_md(melted_df, level, results_dir)

        # Generate project-specific results.md files
        generate_project_specific_results_md(melted_df, level, results_dir)

def generate_top_level_results_md(melted_df, level, results_dir):
    """Generate top-level results.md with best F1-Score results."""
    if 'f1_1' not in melted_df.columns or melted_df['f1_1'].isnull().all():
        logging.warning(f"No 'f1_1' column found or all values are null for level {level}. Cannot generate top-level F1-based report.")
        return

    # Sort by F1-score and get top 10
    top_results = melted_df.nlargest(10, 'f1_1')

    # Prepare table data for top 10
    top_10_table_data = []
    top_10_headers = ['Rank', 'Project', 'Resampling Method', 'ML Algorithm', 'F1-Score (Bug)', 'Accuracy', 'AUC']

    for idx, (_, row) in enumerate(top_results.iterrows(), 1):
        top_10_table_data.append([
            idx,
            row.get('project', 'N/A'),
            row.get('resampling', 'N/A'),
            row.get('model', 'N/A'),
            format_metric(row.get('f1_1')),
            format_metric(row.get('accuracy')),
            format_metric(row.get('auc'))
        ])

    # Get best F1-score result for each project
    best_per_project = melted_df.loc[melted_df.groupby('project')['f1_1'].idxmax()]

    # Prepare table data for best per project
    best_per_project_table_data = []
    best_per_project_headers = ['Project', 'Resampling Method', 'ML Algorithm', 'F1-Score (Bug)', 'Accuracy', 'Precision (Bug)', 'Recall (Bug)', 'AUC', 'MCC']

    for _, row in best_per_project.iterrows():
        best_per_project_table_data.append([
            row.get('project', 'N/A'),
            row.get('resampling', 'N/A'),
            row.get('model', 'N/A'),
            format_metric(row.get('f1_1')),
            format_metric(row.get('accuracy')),
            format_metric(row.get('precision_1')),
            format_metric(row.get('recall_1')),
            format_metric(row.get('auc')),
            format_metric(row.get('mcc'))
        ])

    # Get best result for each ML algorithm based on F1-Score
    best_per_algorithm_table_data = []
    best_per_algorithm_headers = ['ML Algorithm', 'Project', 'Resampling Method', 'F1-Score (Bug)', 'Accuracy', 'Precision (Bug)', 'Recall (Bug)', 'AUC', 'MCC']
    if 'f1_1' in melted_df.columns and not melted_df['f1_1'].isnull().all():
        best_per_algorithm = melted_df.loc[melted_df.groupby('model')['f1_1'].idxmax()]
        # Sort by F1 score descending for presentation
        best_per_algorithm = best_per_algorithm.sort_values(by='f1_1', ascending=False)

        for _, row in best_per_algorithm.iterrows():
            best_per_algorithm_table_data.append([
                row.get('model', 'N/A').replace('_', ' ').title(),
                row.get('project', 'N/A'),
                row.get('resampling', 'N/A'),
                format_metric(row.get('f1_1')),
                format_metric(row.get('accuracy')),
                format_metric(row.get('precision_1')),
                format_metric(row.get('recall_1')),
                format_metric(row.get('auc')),
                format_metric(row.get('mcc'))
            ])

    # Generate markdown
    markdown_content = f"""# {level.title()} Level Analysis - Top Results

## Top 10 Best F1-Score Results

{tabulate.tabulate(top_10_table_data, headers=top_10_headers, tablefmt='pipe')}

## Best F1-Score Result for Each Project

{tabulate.tabulate(best_per_project_table_data, headers=best_per_project_headers, tablefmt='pipe')}
"""

    if best_per_algorithm_table_data:
        markdown_content += f"""
## Best Overall Result for Each ML Algorithm

*The best performing run for each algorithm across all projects and resampling methods, ranked by F1-Score (Bug).*

{tabulate.tabulate(best_per_algorithm_table_data, headers=best_per_algorithm_headers, tablefmt='pipe')}
"""

    markdown_content += f"""
*Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""

    # Save to file
    results_file = results_dir / 'results.md'
    try:
        with open(results_file, 'w', encoding='utf-8') as f:
            f.write(markdown_content)
        logging.info(f"Top-level results saved to {results_file}")
    except Exception as e:
        logging.error(f"Error saving top-level results to {results_file}: {e}")

def generate_project_specific_results_md(melted_df, level, results_dir):
    """Generate project-specific results.md files."""
    if 'f1_1' not in melted_df.columns:
        logging.warning(f"No f1_1 column found for level {level}")
        return

    # Group by project
    for project_name, project_data in melted_df.groupby('project'):
        # Find best result based on f1_1 score
        best_result = project_data.loc[project_data['f1_1'].idxmax()]

        # Prepare table data for best result
        best_headers = ['Project', 'Resampling Method', 'ML Algorithm', 'Accuracy', 'Precision (Bug)', 'Recall (Bug)', 'F1-Score (Bug)', 'AUC', 'MCC']
        best_table_data = [[
            project_name,
            best_result.get('resampling', 'N/A'),
            best_result.get('model', 'N/A'),
            format_metric(best_result.get('accuracy')),
            format_metric(best_result.get('precision_1')),
            format_metric(best_result.get('recall_1')),
            format_metric(best_result.get('f1_1')),
            format_metric(best_result.get('auc')),
            format_metric(best_result.get('mcc'))
        ]]

        # Prepare table data for all resampling methods - show best model for each resampling method
        all_resampling_headers = ['Resampling Method', 'Best ML Algorithm', 'Accuracy', 'Precision (Bug)', 'Recall (Bug)', 'F1-Score (Bug)', 'AUC', 'MCC']
        all_resampling_table_data = []

        # Group by resampling method and find best model for each
        for resampling_method, resampling_data in project_data.groupby('resampling'):
            best_for_resampling = resampling_data.loc[resampling_data['f1_1'].idxmax()]
            all_resampling_table_data.append([
                resampling_method,
                best_for_resampling.get('model', 'N/A'),
                format_metric(best_for_resampling.get('accuracy')),
                format_metric(best_for_resampling.get('precision_1')),
                format_metric(best_for_resampling.get('recall_1')),
                format_metric(best_for_resampling.get('f1_1')),
                format_metric(best_for_resampling.get('auc')),
                format_metric(best_for_resampling.get('mcc'))
            ])

        # Sort by F1-Score (Bug) in descending order
        all_resampling_table_data.sort(key=lambda x: float(x[5]) if x[5] != 'N/A' else 0, reverse=True)

        # Prepare detailed results by ML algorithm and resampling method
        detailed_results_section = generate_detailed_algorithm_results(project_data)

        # Generate markdown
        markdown_content = f"""# {project_name} - {level.title()} Level Analysis

## Best Overall Performance Result

{tabulate.tabulate(best_table_data, headers=best_headers, tablefmt='pipe')}

*Best result selected based on highest F1-Score for bug class*

## Results by Resampling Method

{tabulate.tabulate(all_resampling_table_data, headers=all_resampling_headers, tablefmt='pipe')}

*For each resampling method, the best performing ML algorithm is shown*

{detailed_results_section}

*Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""

        # Create project directory if it doesn't exist
        project_dir = results_dir / project_name
        project_dir.mkdir(parents=True, exist_ok=True)

        # Save to file
        results_file = project_dir / 'results.md'
        try:
            with open(results_file, 'w', encoding='utf-8') as f:
                f.write(markdown_content)
            logging.info(f"Project results saved to {results_file}")
        except Exception as e:
            logging.error(f"Error saving project results to {results_file}: {e}")

def generate_metadata_file():
    """
    Generates a metadata.md file and distribution plots in metadata/figures
    by reading from pre-generated statistics CSV files.
    """
    logging.info("Generating metadata file and figures from statistics...")

    stats_paths = {
        'Commit': RESULTS_COMMIT_LEVEL_DIR / "_statistics" / "commit_dataset_statistics.csv",
        'File': RESULTS_FILE_LEVEL_DIR / "_statistics" / "file_dataset_statistics.csv",
        'Method': RESULTS_METHOD_LEVEL_DIR / "_statistics" / "method_dataset_statistics.csv",
    }

    all_files_exist = True
    for level, path in stats_paths.items():
        if not path.exists():
            logging.error(f"Statistics file not found: {path}. Cannot generate metadata.")
            print(f"Error: Statistics file for {level} level not found at '{path}'.")
            all_files_exist = False
    if not all_files_exist:
        print("\nPlease run the analysis with '--stats_only' for all levels to generate these files first.")
        return

    try:
        metadata_dir = BASE_DIR / 'metadata'
        figures_dir = metadata_dir / 'figures'
        figures_dir.mkdir(parents=True, exist_ok=True)

        for level, path in stats_paths.items():
            if not path.exists():
                continue

            df_stats = pd.read_csv(path)
            # Exclude 'combined' and 'combine' projects from statistics
            df_stats = df_stats[~df_stats['project'].isin(['combined', 'combine'])]
            if df_stats.empty:
                logging.warning(f"Statistics file is empty: {path}. Skipping plot for {level}.")
                continue

            df_stats['total_samples'] = df_stats['bug_samples'] + df_stats['non_bug_samples']
            df_stats = df_stats.sort_values('total_samples', ascending=False)

            plt.style.use('seaborn-v0_8-whitegrid')
            fig, ax = plt.subplots(figsize=(max(12, 0.7 * len(df_stats)), 8))

            projects = df_stats['project']
            x = np.arange(len(projects))
            width = 0.4

            rects1 = ax.bar(x - width/2, df_stats['bug_samples'], width, label='Bugs', color='#d62728')
            rects2 = ax.bar(x + width/2, df_stats['non_bug_samples'], width, label='Clean', color='#1f77b4')

            ax.set_ylabel('Number of Samples', fontsize=12)
            ax.set_title(f'Bugs vs. Clean Samples per Project ({level} Level)', fontsize=16, pad=20)
            ax.set_xticks(x)
            ax.set_xticklabels(projects, rotation=45, ha="right", fontsize=10)
            ax.legend(fontsize=12)

            max_val = df_stats['total_samples'].max()
            if max_val > 5000:
                ax.set_yscale('log')
                ax.set_ylabel('Number of Samples (Log Scale)', fontsize=12)

            fig.tight_layout(pad=1.5)

            figure_path = figures_dir / f"{level.lower()}_bugs_vs_clean.png"
            plt.savefig(figure_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            logging.info(f"Generated distribution plot: {figure_path}")

        num_projects = 0
        total_entries_map = {}
        class_imbalance_map = {}
        nested_pie_data = {'labels': [], 'bugs': [], 'clean': []}

        for level, path in stats_paths.items():
            df = pd.read_csv(path)
            # Exclude 'combined' and 'combine' projects from metadata calculation
            df = df[~df['project'].isin(['combined', 'combine'])]
            if df.empty:
                logging.warning(f"Statistics file is empty after filtering 'combined': {path}. Skipping for metadata calculation.")
                continue

            if num_projects == 0:
                num_projects = len(df['project'].unique())

            total_samples = df['total_samples'].sum()
            total_bugs = df['bug_samples'].sum()
            total_non_bugs = df['non_bug_samples'].sum()

            nested_pie_data['labels'].append(level.title())
            nested_pie_data['bugs'].append(total_bugs)
            nested_pie_data['clean'].append(total_non_bugs)
            total_entries_map[level] = f"{level}: {total_samples:,}"
            imbalance = total_non_bugs / total_bugs if total_bugs > 0 else 0
            class_imbalance_map[level] = f"{level}: {imbalance:.2f}"

        # Generate and save the grouped bar chart if there is data
        if nested_pie_data['labels']:
            labels = nested_pie_data['labels']
            bug_counts = nested_pie_data['bugs']
            clean_counts = nested_pie_data['clean']

            x = np.arange(len(labels))  # the label locations
            width = 0.35  # the width of the bars

            fig, ax = plt.subplots(figsize=(10, 7))
            rects1 = ax.bar(x - width/2, bug_counts, width, label='Bugs', color='#d62728')
            rects2 = ax.bar(x + width/2, clean_counts, width, label='Clean', color='#1f77b4')

            # Add some text for labels, title and axes ticks
            ax.set_ylabel('Number of Samples')
            ax.set_title('Distribution of Buggy and Clean Samples by Granularity')
            ax.set_xticks(x)
            ax.set_xticklabels(labels)
            ax.legend()

            def autolabel(rects):
                """Attach a text label above each bar in *rects*, displaying its height."""
                for rect in rects:
                    height = rect.get_height()
                    ax.annotate(f'{height:,}',
                                xy=(rect.get_x() + rect.get_width() / 2, height),
                                xytext=(0, 3),  # 3 points vertical offset
                                textcoords="offset points",
                                ha='center', va='bottom')

            autolabel(rects1)
            autolabel(rects2)

            fig.tight_layout()

            chart_path = figures_dir / "total_entries_distribution.png"
            plt.savefig(chart_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            logging.info(f"Total entries distribution bar chart saved to {chart_path}")

        total_entries_str = " <br> ".join(total_entries_map.values())
        class_imbalance_str = " <br> ".join(class_imbalance_map.values())

        metadata_file = metadata_dir / 'metadata.md'

        table_content = f"""
| Attribute               | Value                                                 |
|-------------------------|-------------------------------------------------------|
| Granularity             | Commit, File, Method                                  |
| Number of Projects      | {num_projects}                                        |
| Total Entries           | {total_entries_str}                                   |
| Overall Class Imbalance | {class_imbalance_str}                                 |
"""

        full_content = f"""# Project Metadata

{table_content.strip()}

*Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""

        with open(metadata_file, 'w', encoding='utf-8') as f:
            f.write(full_content)
        logging.info(f"Metadata file saved to {metadata_file}")
        print(f"Metadata file saved to {metadata_file}")
    except Exception as e:
        logging.error(f"Error during metadata generation: {e}", exc_info=True)


def regenerate_all_figures(args, level_data_dir, projects_to_analyze, resampling_paths_to_iterate):
    """Regenerate figures from existing analysis results."""
    logging.info(f"Regenerating figures for level {args.level}")
    # This function would load existing ROC data and regenerate plots
    # For now, it's a placeholder
    logging.warning("regenerate_all_figures is not fully implemented yet.")

def main():
    parser = argparse.ArgumentParser(
        description='Run bug prediction analysis for different granularity levels.',
        epilog='''
Results Directory Structure:
  results_{level}_level/
    └── {project}/
        └── {cv_type}/           # temporal | shuffle
            └── {feature_set}/   # full | no_go_metrics
                └── {resampling}/ # none | smote | adasyn | ...
                    ├── *_fold_metrics.json
                    └── analysis_summary.json

Example Usage:
  # Temporal CV with full features and no resampling
  python analiz.py --project influxdb --level method --resampling none

  # Shuffle CV with Go metrics excluded and SMOTE resampling
  python analiz.py --project influxdb --level method --resampling smote --shuffle-cv --exclude-go-metrics

  # Compare Go metrics impact
  python analiz.py --level method --wilcoxon-go-metrics --resampling none
        ''',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # General arguments
    parser.add_argument('--level', type=str, required=False, choices=ALL_LEVELS,
                        help='The granularity level to analyze (commit, file, method). Required for standard analysis.')
    parser.add_argument('--project', type=str, default='all',
                        help='The project to analyze. Use "all" to run for all projects at the specified level.')
    parser.add_argument('--resampling', type=str, default='none',
                        choices=ALL_ACTUAL_RESAMPLING_METHODS + ['none', 'all'],
                        help='Resampling strategy to balance data. "all" to run for all strategies.')
    parser.add_argument('--folds', type=int, default=5,
                        help='Number of folds for cross-validation.')
    parser.add_argument('--optimize', action='store_true',
                        help='Perform hyperparameter optimization for models.')
    parser.add_argument('--methods', nargs='+', default='all',
                        choices=ALL_CLASSIFIER_FUNCTION_NAMES + ['all'],
                        help='Specify which ML algorithms to run.')
    parser.add_argument('--overlap-only', action='store_true',
                        help='Use only the overlapping time period where both bugs and non-bugs exist. '
                             'This helps with temporal CV when non-bug data collection started later than bug data.')
    parser.add_argument('--min-class-ratio', type=float, default=0.05,
                        help='Minimum minority class ratio required in train/test folds. '
                             'Folds below this threshold are skipped. Default: 0.05 (5%%). '
                             'Set to 0 to disable this check.')
    parser.add_argument('--shuffle-cv', action='store_true',
                        help='Use stratified shuffle CV instead of temporal CV. '
                             'Results saved to: {project}/shuffle/{feature_set}/{resampling}/')

    # Feature Selection Arguments
    parser.add_argument('--select-feature', type=str, default=None,
                        choices=['variance', 'chi2', 'rfe', 'lasso', 'rf', 'mi', 'combine'],
                        help='Dynamic feature selection method to use.')
    parser.add_argument('--k-features', type=int, default=None,
                        help='Number of top features to select. If not specified, auto-determined based on '
                             'Guyon & Elisseeff (2003): small (≤10): n-1, medium (11-30): 70%%, large (>30): 2*sqrt(n)')

    # Class Imbalance Handling Arguments
    parser.add_argument('--class-weight', type=str, default='auto',
                        choices=['auto', 'balanced', 'none'],
                        help="Class weight strategy. 'auto': disable when resampling is used (default). "
                             "'balanced': always use balanced class weights. "
                             "'none': never use class weights.")

    # Reporting and Utility Arguments
    parser.add_argument('--stats_only', action='store_true',
                        help='Generate and print dataset statistics for all levels (if --level is not specified).')
    parser.add_argument('--generate-tables', action='store_true',
                        help='Generate all summary markdown tables for the specified level. '
                             'Uses --shuffle-cv and --exclude-go-metrics to select which results to summarize.')
    parser.add_argument('--generate-summary-plots', action='store_true',
                        help='Generate summary bar charts from existing results.md files.')
    parser.add_argument('--regenerate-figures', action='store_true',
                        help='Regenerate all ROC and metric figures from existing JSON results.')
    parser.add_argument('--generate-reports', action='store_true',
                        help='Generate comprehensive reports (results.md + plots) for each CV type (temporal/shuffle) '
                             'including holdout results, CV metrics with std, and comparison charts.')
    parser.add_argument('--friedman-test', action='store_true', help='Run Friedman test on results.')
    parser.add_argument('--nemenyi-test', action='store_true', help='Run Nemenyi post-hoc test on results.')
    parser.add_argument('--metadata', action='store_true',
                        help='Generate a metadata.md file in a metadata directory.')

    parser.add_argument('--exclude-go-metrics', action='store_true',
                        help='Exclude Go-specific metrics from the analysis.')

    # Statistical comparison of Go-metrics impact
    parser.add_argument('--wilcoxon-go-metrics', action='store_true',
                        help='Run Wilcoxon signed-rank test to compare results with and without Go-specific metrics.')

    # New argument for generating feature importance table
    parser.add_argument('--important-features', action='store_true',
                        help='Generate and append the aggregate feature importance table for the combined dataset at the specified level.')
    parser.add_argument('--find-and-optimize', action='store_true',
                        help='Sequentially run best feature selection and then hyperparameter optimization with the found features.')
    parser.add_argument('--best-features', action='store_true',
                        help='Find the best number of features by iterating and evaluating model performance.')

    # CPDP Arguments
    cpdp_group = parser.add_argument_group('Cross-Project Defect Prediction (CPDP)')
    cpdp_group.add_argument('--cpdp', action='store_true',
                        help='Enable Cross-Project Defect Prediction mode.')
    cpdp_group.add_argument('--lopo', action='store_true',
                        help='Enable Leave-One-Project-Out (LOPO) cross-validation mode.')
    cpdp_group.add_argument('--source', type=str,
                        help='Source project for training in CPDP mode.')
    cpdp_group.add_argument('--destination', type=str,
                        help='Destination project for testing in CPDP mode.')

    # Instance Selection Arguments
    instance_selection_group = parser.add_argument_group('Instance Selection (for CPDP/LOPO)')
    instance_selection_group.add_argument('--instance-selection', type=str, choices=['nn_filter'],
                                          help='Instance selection method to filter source data.')
    instance_selection_group.add_argument('--k-neighbors', type=int, default=10,
                                          help='Number of neighbors (k) for nn_filter.')

    # Bootstrap CI Arguments
    bootstrap_group = parser.add_argument_group('Bootstrap Confidence Intervals')
    bootstrap_group.add_argument('--bootstrap-ci', action='store_true',
                                 help='Compute bootstrap confidence intervals for holdout metrics.')
    bootstrap_group.add_argument('--n-bootstrap', type=int, default=1000,
                                 help='Number of bootstrap iterations (default: 1000).')

    cli_args = parser.parse_args()

    # Set global class weight mode from CLI argument
    global _CLASS_WEIGHT_MODE
    _CLASS_WEIGHT_MODE = cli_args.class_weight
    logging.info(f"Class weight mode set to: '{_CLASS_WEIGHT_MODE}'")

    # Handle CPDP mode first, as it has a different workflow
    if cli_args.cpdp:
        if not cli_args.source or not cli_args.destination or not cli_args.level:
            parser.error("--cpdp mode requires --level, --source, and --destination.")
        run_cpdp_workflow(cli_args)
        return

    if cli_args.lopo:
        if not cli_args.level:
            parser.error("--lopo mode requires --level.")
        run_lopo_workflow(cli_args)
        return

    # Handle single-action arguments first
    if cli_args.metadata:
        generate_metadata_file()
        return

    if cli_args.find_and_optimize:
        if not cli_args.level:
            parser.error("--level is required for --find-and-optimize.")

        level_data_dir = get_data_dir(cli_args.level)
        projects_to_run = []
        if cli_args.project == 'all':
            projects_to_run = [d.name for d in level_data_dir.iterdir() if d.is_dir() and not d.name.startswith('_')]
            logging.info(f"Starting --find-and-optimize for all projects: {projects_to_run}")
        else:
            projects_to_run.append(cli_args.project)

        resampling_to_run = []
        if cli_args.resampling == 'all':
            resampling_to_run = ['none'] + ALL_ACTUAL_RESAMPLING_METHODS
            logging.info(f"Starting --find-and-optimize for all resampling methods.")
        else:
            resampling_to_run.append(cli_args.resampling)

        logging.info("--- Batch Find and Optimize Workflow Started ---")

        for proj_name in projects_to_run:
            for resample_name in resampling_to_run:
                logging.info(f"--- Running for Project: '{proj_name}', Resampling: '{resample_name}' ---")

                temp_args = argparse.Namespace(**vars(cli_args))
                temp_args.project = proj_name
                temp_args.resampling = resample_name

                logging.info(f"Step 1: Finding the best feature set...")
                best_features_found = find_best_feature_set(temp_args)

                if not best_features_found:
                    logging.error(f"Could not determine the best feature set for {proj_name} with {resample_name}. Skipping to next combination.")
                    continue

                logging.info(f"Step 1 Complete. Found {len(best_features_found)} best features: {best_features_found}")
                logging.info("---")
                logging.info(f"Step 2: Running hyperparameter optimization with the best feature set...")

                analyze_project(
                    project_name=proj_name,
                    level=cli_args.level,
                    resampling_strategy=resample_name if resample_name != 'none' else None,
                    n_folds=cli_args.folds,
                    optimize=True, # Force optimization
                    selected_features_config=best_features_found,
                    methods_to_run=cli_args.methods,
                    cli_args=temp_args,
                    progress=None # Can't easily pass progress bar here, running standalone.
                )
                logging.info(f"--- Finished for Project: '{proj_name}', Resampling: '{resample_name}' ---")

        logging.info("--- Batch Find and Optimize Workflow Complete ---")
        return

    if cli_args.generate_summary_plots:
        levels_to_process_plots = [cli_args.level] if cli_args.level else ALL_LEVELS
        if not cli_args.level:
            logging.info("`--level` not specified, running plot generation for all levels.")
        for level in levels_to_process_plots:
            logging.info(f"Generating summary plots for level: {level}")
            results_dir = get_results_dir(level)

            # For second request: Best F1 score per project for a given level
            plot_best_f1_per_project(level, results_dir)

            # For first request: F1 scores per resampling method for each project
            plot_f1_scores_per_resampling(level, results_dir)

            # For third request: F1 scores by ML algorithm for each project
            plot_f1_scores_by_ml_algorithm(level, results_dir)
        return

    if cli_args.important_features:
        if cli_args.level:
            generate_important_features_table(cli_args.level)
        else:
            print("Error: --level is required when using --important-features.")
        return

    # This check for stats was here, but it's better handled with the other utility modes below.
    # The logic is preserved and consolidated.

    # Validate --level requirement and handle different run modes.
    is_statistical_test = cli_args.friedman_test or cli_args.nemenyi_test or cli_args.wilcoxon_go_metrics
    is_utility_mode = cli_args.regenerate_figures or cli_args.generate_tables or cli_args.stats_only or is_statistical_test or cli_args.important_features or cli_args.metadata or cli_args.best_features or cli_args.generate_summary_plots or cli_args.generate_reports

    # If not in a specific utility mode and no level is provided, it's an error.
    if not is_utility_mode and not cli_args.level:
        parser.error("--level is required for a standard analysis run. Or, specify a utility mode like --stats_only, --regenerate-figures, etc.")

    if cli_args.regenerate_figures:
        logging.info(f"--regenerate-figures specified. Figures will be generated for level: {cli_args.level or 'ALL'}")
    if cli_args.generate_tables:
        logging.info(f"--generate-tables specified. Tables will be generated for level: {cli_args.level or 'ALL'}")
    if cli_args.stats_only:
        logging.info(f"--stats_only specified. Statistics will be collected for level: {cli_args.level or 'ALL'}")

    # Handle statistical test generation first if specified
    if cli_args.friedman_test:
        if not cli_args.level:
            parser.error("--level is required for statistical tests.")
        if not cli_args.resampling or cli_args.resampling == 'none':
            logging.info("Resampling strategy not specified for Friedman test; defaulting to run on 'all' available strategies.")
            cli_args.resampling = 'all' # Default to all resampling strategies
        run_friedman_analysis(cli_args)
        return # Exit after statistical analysis
    elif cli_args.nemenyi_test:
        if not cli_args.level:
            parser.error("--level is required for statistical tests.")
        if not cli_args.resampling or cli_args.resampling == 'none':
            logging.info("Resampling strategy not specified for Nemenyi test; defaulting to run on 'all' available strategies.")
            cli_args.resampling = 'all'
        run_nemenyi_analysis(cli_args)
        return # Exit after statistical analysis
    elif cli_args.wilcoxon_go_metrics:
        if not cli_args.level:
            parser.error("--level is required for Wilcoxon Go-metrics test.")
        if not cli_args.resampling or cli_args.resampling == 'none':
            logging.info("Resampling strategy not specified for Wilcoxon test; defaulting to 'all' available strategies.")
            cli_args.resampling = 'all'
        run_wilcoxon_go_metrics_analysis(cli_args)
        return # Exit after statistical analysis

    if cli_args.regenerate_figures:
        levels_to_process_figures = [cli_args.level] if cli_args.level else ALL_LEVELS
        for current_level_for_figures in levels_to_process_figures:
            logging.info(f"===== Regenerating figures for LEVEL: {current_level_for_figures} =====")
            level_data_dir_fig = get_data_dir(current_level_for_figures)
            projects_to_analyze_fig = []
            if cli_args.project == 'all':
                projects_to_analyze_fig = [d.name for d in level_data_dir_fig.iterdir() if d.is_dir()] if level_data_dir_fig.exists() else []
            else:
                projects_to_analyze_fig.append(cli_args.project)

            if not projects_to_analyze_fig:
                logging.warning(f"No projects found for level {current_level_for_figures} at {level_data_dir_fig}. Skipping figure generation for this level.")
                continue

            # Determine resampling strategy paths for figure regeneration
            figure_resampling_paths_to_iterate = []
            if cli_args.resampling == 'all':
                logging.info(f"--regenerate-figures mode with --resampling=all. Will iterate over 'none' and all {len(ALL_ACTUAL_RESAMPLING_METHODS)} actual resampling methods.")
                figure_resampling_paths_to_iterate = ['none'] + ALL_ACTUAL_RESAMPLING_METHODS
            elif cli_args.resampling == 'none' or not cli_args.resampling:
                logging.info(f"--regenerate-figures mode with --resampling=none or not set. Will iterate over 'none' and all {len(ALL_ACTUAL_RESAMPLING_METHODS)} actual resampling methods to find existing results.")
                figure_resampling_paths_to_iterate = ['none'] + ALL_ACTUAL_RESAMPLING_METHODS
            else: # A specific resampling method like 'smote' was given
                logging.info(f"--regenerate-figures mode with specific resampling: {cli_args.resampling}. Will iterate only for this method.")
                figure_resampling_paths_to_iterate = [cli_args.resampling]

            temp_args_for_level = argparse.Namespace(**vars(cli_args))
            temp_args_for_level.level = current_level_for_figures

            regenerate_all_figures(temp_args_for_level, level_data_dir_fig, projects_to_analyze_fig, figure_resampling_paths_to_iterate)
        return

    if cli_args.generate_reports:
        run_generate_reports(cli_args)
        return

    if cli_args.generate_tables:
        levels_to_process_results = [cli_args.level] if cli_args.level else ALL_LEVELS
        project_filter = cli_args.project if cli_args.project != 'all' else None
        generate_markdown_tables(levels_to_process_results, project_filter, cli_args)
        return

    if cli_args.stats_only:
        # Process statistics for specified level(s)
        levels_to_process_stats = [cli_args.level] if cli_args.level else ALL_LEVELS

        for current_level in levels_to_process_stats:
            logging.info(f"===== Collecting statistics for LEVEL: {current_level} =====")
            level_data_dir = get_data_dir(current_level)

            if not level_data_dir.exists():
                logging.warning(f"Data directory not found for level {current_level}: {level_data_dir}")
                continue

            all_stats_data = []
            if cli_args.project == 'all':
                project_names_to_stat = [d.name for d in level_data_dir.iterdir() if d.is_dir()]
            else:
                project_names_to_stat = [cli_args.project]

            for proj_name in project_names_to_stat:
                stats = collect_dataset_statistics(proj_name, current_level)
                if stats: all_stats_data.append(stats)

            if all_stats_data:
                stats_df = pd.DataFrame(all_stats_data)
                logging.info(f"\nDataset Statistics for {current_level} level:\n" + stats_df.to_string())
                # Save stats
                stats_output_dir = get_results_dir(current_level) / "_statistics"
                stats_output_dir.mkdir(parents=True, exist_ok=True)
                stats_df.to_csv(stats_output_dir / f"{current_level}_dataset_statistics.csv", index=False)
                logging.info(f"Dataset statistics saved to {stats_output_dir}")
            else:
                logging.info(f"No statistics collected for level {current_level}.")
        return

    # This part onwards is for the main analysis run
    if not cli_args.level:
         parser.error("--level is required for a standard analysis run.")

    level_data_dir = get_data_dir(cli_args.level)
    projects_to_analyze = []

    logging.info(f"Level data directory: {level_data_dir}")
    logging.info(f"Project parameter: '{cli_args.project}'")

    if cli_args.project == 'all':
        available_projects = [d.name for d in level_data_dir.iterdir() if d.is_dir()]
        projects_to_analyze = available_projects
        logging.info(f"Project set to 'all'. Found {len(available_projects)} projects: {available_projects}")
    else:
        projects_to_analyze.append(cli_args.project)
        logging.info(f"Single project specified: '{cli_args.project}'")

        # Check if the project directory exists
        project_dir = level_data_dir / cli_args.project
        if not project_dir.exists():
            logging.warning(f"Project directory does not exist: {project_dir}")
        elif not project_dir.is_dir():
            logging.warning(f"Project path exists but is not a directory: {project_dir}")
        else:
            logging.info(f"Project directory confirmed: {project_dir}")

    logging.info(f"Final projects to analyze: {projects_to_analyze}")

    if not projects_to_analyze:
        logging.error(f"No projects found to analyze for level {cli_args.level} in {level_data_dir}")
        return

    # Determine which resampling strategies to run for analysis
    # strategies_to_run_paths will contain strings like 'none', 'smote', etc.
    if cli_args.resampling == 'all':
        strategies_to_run_paths = ALL_ACTUAL_RESAMPLING_METHODS
        logging.info(f"'--resampling all' specified. Running all strategies except 'none': {strategies_to_run_paths}")
    else: # This handles the default ('none') and specific methods like 'smote'
        strategies_to_run_paths = [cli_args.resampling]
        if cli_args.resampling == 'none':
            logging.info("No resampling strategy specified or '--resampling none' used. Running analysis only without resampling.")
        else:
            logging.info(f"Specific resampling strategy specified: {strategies_to_run_paths}")

    logging.info(f"Will run analysis for {len(strategies_to_run_paths)} resampling strategies: {strategies_to_run_paths}")

    progress_columns = [
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        MofNCompleteColumn(),
        TimeRemainingColumn(),
        TimeElapsedColumn(),
    ]
    with Progress(*progress_columns, transient=True) as progress:
        resampling_task_id = progress.add_task("[bold cyan]Resampling Strategies", total=len(strategies_to_run_paths))

        for path_strategy_name in strategies_to_run_paths:
            # Determine the actual strategy to pass to analyze_project for resampling logic
            # This will be None if path_strategy_name is 'none', otherwise the method name string.
            actual_logic_strategy = None if path_strategy_name == 'none' else path_strategy_name
            # strategy_name_for_path_and_log is already path_strategy_name (string)
            logging.info(f"===== Starting analysis for resampling strategy: {path_strategy_name} =====")

            current_strategy_overall_results = []

            project_task_id = progress.add_task(f"[green]Projects ({path_strategy_name})", total=len(projects_to_analyze))

            for proj_name in projects_to_analyze:
                if not (level_data_dir / proj_name).is_dir():
                    logging.warning(f"Project directory {level_data_dir / proj_name} not found. Skipping {proj_name}.")
                    progress.update(project_task_id, advance=1)
                    continue

                logging.info(f"--- Analyzing project: {proj_name} with strategy: {path_strategy_name} ---")
                # Handle methods_to_run parameter
                methods_to_run = cli_args.methods
                if isinstance(methods_to_run, list) and 'all' in methods_to_run:
                    methods_to_run = 'all'

                # Check for bootstrap CI flags
                compute_bootstrap_ci = getattr(cli_args, 'bootstrap_ci', False)
                n_bootstrap = getattr(cli_args, 'n_bootstrap', 1000)

                project_results = analyze_project(
                    project_name=proj_name,
                    level=cli_args.level,
                    resampling_strategy=actual_logic_strategy, # Pass None or method name string
                    n_folds=cli_args.folds,
                    optimize=cli_args.optimize,
                    selected_features_config=None,  # This parameter is not used with dynamic feature selection
                    methods_to_run=methods_to_run,
                    cli_args=cli_args, # Pass the full cli_args object
                    progress=progress,
                    project_task_id=project_task_id,
                    compute_bootstrap_ci=compute_bootstrap_ci,
                    n_bootstrap=n_bootstrap
                )
                if project_results:
                    print_final_metrics(project_results)
                    current_strategy_overall_results.append(project_results)

                progress.update(project_task_id, advance=1)

            progress.remove_task(project_task_id)
            progress.update(resampling_task_id, advance=1)

            if current_strategy_overall_results:
                rows_for_summary_df = []
                for res_dict in current_strategy_overall_results:
                    flat_res_dict = {
                        'project': res_dict.get('project'),
                        'level': res_dict.get('level'),
                        'resampling': res_dict.get('resampling'), # This is set by analyze_project
                        'cv_type': get_cv_type_name(cli_args.shuffle_cv if hasattr(cli_args, 'shuffle_cv') else False),
                        'feature_set': get_feature_set_name(cli_args.exclude_go_metrics if hasattr(cli_args, 'exclude_go_metrics') else False)
                    }
                    for model_name, metrics_val in res_dict.items():
                        if model_name not in ['project', 'level', 'resampling', 'cv_type', 'feature_set']:
                            if isinstance(metrics_val, dict):  # metrics_val is the avg_metrics dict
                                for metric_key, specific_metric_val in metrics_val.items():
                                    if metric_key != 'feature_importance':
                                         # Ensure the metric is scalar before adding
                                        if not isinstance(specific_metric_val, (list, tuple, dict, np.ndarray)):
                                            flat_res_dict[f'{model_name}_{metric_key}'] = specific_metric_val
                                    elif metric_key == 'error' and isinstance(specific_metric_val, str): # Capture error messages for models
                                        flat_res_dict[f'{model_name}_error'] = specific_metric_val
                    rows_for_summary_df.append(flat_res_dict)

                summary_df = pd.DataFrame(rows_for_summary_df)

                # New hierarchical summary structure: _summary/{cv_type}/{feature_set}/{resampling}/
                cv_type = get_cv_type_name(cli_args.shuffle_cv if hasattr(cli_args, 'shuffle_cv') else False)
                feature_set = get_feature_set_name(cli_args.exclude_go_metrics if hasattr(cli_args, 'exclude_go_metrics') else False)
                summary_dir = get_results_dir(cli_args.level) / "_summary" / cv_type / feature_set / path_strategy_name
                summary_dir.mkdir(parents=True, exist_ok=True)

                project_name_in_file = cli_args.project if cli_args.project != 'all' or len(projects_to_analyze) == 1 else 'all_projects'

                summary_file_path = summary_dir / f"classification_summary_{cli_args.level}_{project_name_in_file}_{path_strategy_name}.csv"
                summary_df.to_csv(summary_file_path, index=False)
                logging.info(f"Aggregated analysis results for strategy '{path_strategy_name}' saved to {summary_file_path}")
                logging.info(f"\\nAggregated Results Summary for strategy '{path_strategy_name}':\\n" + summary_df.to_string())
            else:
                logging.info(f"No analysis results to summarize for strategy '{path_strategy_name}'.")

    # Call markdown generation (simplified) - Commented out as it may need adaptation for multiple strategies
    # generate_markdown_tables(cli_args.level, project_name_filter=cli_args.project if cli_args.project != 'all' else None, resampling_filter=cli_args.resampling)

    # The original summarization for a single run is now handled within the loop.
    # logging.info("No analysis results to summarize.") # This else is no longer needed in the same way.


def regenerate_all_figures(cli_args, level_data_dir_base, projects_to_run, resampling_strategies_to_run):
    """Regenerates figures from saved classification summary CSV files."""
    logging.info("Starting figure regeneration process...")

    cv_type = get_cv_type_name(cli_args.shuffle_cv if hasattr(cli_args, 'shuffle_cv') else False)
    feature_set = get_feature_set_name(cli_args.exclude_go_metrics if hasattr(cli_args, 'exclude_go_metrics') else False)

    for strategy_name in resampling_strategies_to_run:
        actual_strategy_name = strategy_name if strategy_name is not None else 'none'
        logging.info(f"-- Regenerating figures for resampling strategy: {actual_strategy_name} --")
        # New hierarchical structure
        summary_dir_for_strategy = get_results_dir(cli_args.level) / "_summary" / cv_type / feature_set / actual_strategy_name

        for project_name in projects_to_run:
            logging.info(f"---- Project: {project_name} ----")
            # Construct the expected summary file based on how it's saved in main()
            project_name_in_file = project_name if cli_args.project != 'all' or len(projects_to_run) == 1 else 'all_projects'
            summary_file_name = f"classification_summary_{cli_args.level}_{project_name_in_file}_{actual_strategy_name}.csv"
            summary_file_path = summary_dir_for_strategy / summary_file_name

            if not summary_file_path.exists():
                logging.warning(f"Summary CSV not found, cannot generate figures: {summary_file_path}")
                continue

            try:
                summary_df = pd.read_csv(summary_file_path)
                if summary_df.empty:
                    logging.warning(f"Summary CSV is empty: {summary_file_path}")
                    continue
            except Exception as e:
                logging.error(f"Error reading summary CSV {summary_file_path}: {e}")
                continue

            # Determine the correct output directory for figures for this specific project and strategy
            # This should match the output_dir structure used in analyze_project
            project_results_base_dir = get_results_dir(cli_args.level) / project_name
            analysis_subdir_name = f"analysis_{actual_strategy_name}"
            figure_output_dir = project_results_base_dir / analysis_subdir_name
            figure_output_dir.mkdir(parents=True, exist_ok=True)

            # Reconstruct the 'results' dictionary structure expected by plotting functions
            # This is a simplified reconstruction. ROC data is not directly in the summary CSV.
            # For ROC, we might need to store y_test, y_prob per fold if full regeneration is needed,
            # or accept that ROC curves cannot be perfectly regenerated from summary CSVs alone.

            # For bar charts (F1, Accuracy), we can extract data from the summary_df
            # The summary_df has columns like 'naive_bayes_f1_1', 'naive_bayes_accuracy', etc.
            # We need to transform this back into a nested dict like `results` in `analyze_project`

            # Assuming one row per project in the summary_df if project_name_in_file was specific.
            # If 'all_projects', we need to filter the row for the current project_name.
            project_summary_data = None
            if project_name_in_file == 'all_projects':
                # The summary CSV for 'all_projects' should have a 'project' column
                if 'project' in summary_df.columns:
                    project_row = summary_df[summary_df['project'] == project_name]
                    if not project_row.empty:
                        project_summary_data = project_row.iloc[0].to_dict()
            else: # Specific project summary file
                project_summary_data = summary_df.iloc[0].to_dict()

            if not project_summary_data:
                logging.warning(f"No data found for project {project_name} in {summary_file_path}")
                continue

            # Reconstruct results dict for plotting bar charts
            reconstructed_results_for_plot = {
                'project': project_name,
                'level': cli_args.level,
                'resampling': actual_strategy_name
            }

            # Extract model metrics from the flattened summary_data
            # Example: project_summary_data might have 'naive_bayes_f1_1', 'naive_bayes_accuracy'
            # We need to group them back: results['naive_bayes'] = {'f1_1': ..., 'accuracy': ...}
            available_models = list(set([col.split('_')[0] for col in project_summary_data.keys() if '_f1_1' in col or '_accuracy' in col]))

            for model_key_name in ALL_CLASSIFIER_FUNCTION_NAMES: # Assuming ALL_CLASSIFIER_FUNCTION_NAMES is defined globally
                f1_col = f'{model_key_name}_f1_1'
                acc_col = f'{model_key_name}_accuracy'
                auc_col = f'{model_key_name}_auc' # and other metrics as needed
                if f1_col in project_summary_data and acc_col in project_summary_data:
                    reconstructed_results_for_plot[model_key_name] = {
                        'f1_1': project_summary_data.get(f1_col),
                        'accuracy': project_summary_data.get(acc_col),
                        'auc': project_summary_data.get(auc_col) # Add other metrics if they are in summary and needed
                    }

            if len(reconstructed_results_for_plot) <= 3: # Only project, level, resampling keys
                logging.warning(f"No model metrics could be reconstructed for {project_name} from {summary_file_path}. Skipping bar charts.")
            else:
                plot_scores_barchart(reconstructed_results_for_plot, 'f1_1', 'F1 Scores (Class 1)',
                                     f'{project_name}_{cli_args.level}_f1_scores', project_name, cli_args.level, figure_output_dir)
                plot_scores_barchart(reconstructed_results_for_plot, 'accuracy', 'Accuracy Scores',
                                     f'{project_name}_{cli_args.level}_accuracy_scores', project_name, cli_args.level, figure_output_dir)

            # ROC curve regeneration from saved JSON
            roc_data_json_path = figure_output_dir / "roc_fold_data.json"
            if roc_data_json_path.exists():
                try:
                    with open(roc_data_json_path, 'r') as f:
                        loaded_roc_data_from_json = json.load(f)

                    if loaded_roc_data_from_json:
                        # Convert lists back to NumPy arrays for sklearn.metrics.roc_curve compatibility
                        # plot_roc_curves expects a structure like:
                        # { 'model_name': [ {'y_test': np.array, 'y_prob': np.array}, ... ], ... }
                        roc_data_for_plotting = {}
                        for model_name, fold_data_list in loaded_roc_data_from_json.items():
                            roc_data_for_plotting[model_name] = []
                            for fold_data in fold_data_list:
                                if isinstance(fold_data, dict) and 'y_test' in fold_data and 'y_prob' in fold_data:
                                    try:
                                        y_test_np = np.array(fold_data['y_test'])
                                        y_prob_np = np.array(fold_data['y_prob'])
                                        # Basic validation of content
                                        if y_test_np.ndim == 1 and y_prob_np.ndim == 1 and len(y_test_np) == len(y_prob_np) and len(y_test_np) > 0:
                                            roc_data_for_plotting[model_name].append({'y_test': y_test_np, 'y_prob': y_prob_np})
                                        else:
                                            logging.warning(f"Skipping invalid or empty fold ROC data for {model_name} in {project_name} from {roc_data_json_path}.")
                                    except Exception as e_conversion:
                                        logging.error(f"Error converting fold ROC data for {model_name} in {project_name} from {roc_data_json_path}: {e_conversion}")
                                else:
                                    logging.warning(f"Skipping malformed fold ROC data entry for {model_name} in {project_name} from {roc_data_json_path}.")

                        if any(roc_data_for_plotting.values()): # Check if any model has valid data
                            plot_roc_curves(roc_data_for_plotting, project_name, cli_args.level, figure_output_dir)
                            logging.info(f"ROC curves regenerated from {roc_data_json_path} for {project_name} ({actual_strategy_name}).")
                        else:
                            logging.warning(f"No valid ROC data to plot after processing {roc_data_json_path} for {project_name} ({actual_strategy_name}).")
                    else:
                        logging.warning(f"Loaded ROC data from {roc_data_json_path} is empty for {project_name} ({actual_strategy_name}).")
                except Exception as e:
                    logging.error(f"Error loading or plotting ROC data from {roc_data_json_path} for {project_name} ({actual_strategy_name}): {e}")
            else:
                logging.warning(f"ROC fold data JSON ({roc_data_json_path}) not found. Cannot regenerate ROC curves for {project_name} ({actual_strategy_name}).")

            # Feature correlation plot regeneration
            # This requires access to the original X data, which is not in the summary CSV.
            # To regenerate this, analyze_project would need to save X or its correlation matrix.
            # For now, skipping.
            logging.warning(f"Feature correlation plot regeneration requires original data and is not supported from summary CSV for {project_name} ({actual_strategy_name}).")

    logging.info("Figure regeneration process completed.")

def generate_detailed_algorithm_results(project_data):
    """Generate detailed results showing each ML algorithm's performance with each resampling method."""

    # Get unique algorithms and resampling methods
    algorithms = sorted(project_data['model'].unique())
    resampling_methods = sorted(project_data['resampling'].unique())

    detailed_section = "\n## Detailed Results by ML Algorithm\n\n"
    detailed_section += "*Performance of each ML algorithm across all resampling methods*\n\n"

    for algorithm in algorithms:
        algorithm_data = project_data[project_data['model'] == algorithm]

        if len(algorithm_data) == 0:
            continue

        detailed_section += f"### {algorithm.replace('_', ' ').title()}\n\n"

        # Create table for this algorithm
        headers = ['Resampling Method', 'Accuracy', 'Precision (Bug)', 'Recall (Bug)', 'F1-Score (Bug)', 'AUC', 'MCC']
        table_data = []

        for resampling_method in resampling_methods:
            method_data = algorithm_data[algorithm_data['resampling'] == resampling_method]

            if len(method_data) > 0:
                # Take the first (and should be only) result for this algorithm-resampling combination
                result = method_data.iloc[0]
                table_data.append([
                    resampling_method,
                    format_metric(result.get('accuracy')),
                    format_metric(result.get('precision_1')),
                    format_metric(result.get('recall_1')),
                    format_metric(result.get('f1_1')),
                    format_metric(result.get('auc')),
                    format_metric(result.get('mcc'))
                ])
            else:
                table_data.append([
                    resampling_method,
                    'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A'
                ])

        # Sort by F1-Score (Bug) in descending order
        table_data.sort(key=lambda x: float(x[4]) if x[4] != 'N/A' else -1, reverse=True)

        detailed_section += tabulate.tabulate(table_data, headers=headers, tablefmt='pipe') + "\n\n"

    # Add summary comparison table - best result for each algorithm
    detailed_section += "## Algorithm Performance Summary\n\n"
    detailed_section += "*Best result for each ML algorithm across all resampling methods*\n\n"

    summary_headers = ['ML Algorithm', 'Best Resampling Method', 'Accuracy', 'Precision (Bug)', 'Recall (Bug)', 'F1-Score (Bug)', 'AUC', 'MCC']
    summary_table_data = []

    for algorithm in algorithms:
        algorithm_data = project_data[project_data['model'] == algorithm]
        if len(algorithm_data) > 0:
            # Handle NaN values in f1_1 column
            valid_f1_data = algorithm_data.dropna(subset=['f1_1'])
            if len(valid_f1_data) > 0:
                best_result = valid_f1_data.loc[valid_f1_data['f1_1'].idxmax()]
                summary_table_data.append([
                    algorithm.replace('_', ' ').title(),
                    best_result.get('resampling', 'N/A'),
                    format_metric(best_result.get('accuracy')),
                    format_metric(best_result.get('precision_1')),
                    format_metric(best_result.get('recall_1')),
                    format_metric(best_result.get('f1_1')),
                    format_metric(best_result.get('auc')),
                    format_metric(best_result.get('mcc'))
                ])
            else:
                # All f1_1 values are NaN for this algorithm
                summary_table_data.append([
                    algorithm.replace('_', ' ').title(),
                    'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A'
                ])

    # Sort by F1-Score (Bug) in descending order
    summary_table_data.sort(key=lambda x: float(x[5]) if x[5] != 'N/A' else -1, reverse=True)

    detailed_section += tabulate.tabulate(summary_table_data, headers=summary_headers, tablefmt='pipe') + "\n\n"

    return detailed_section

def run_friedman_analysis(args):
    """Performs Friedman test for specified levels and resampling strategies."""
    logging.info("Starting Friedman test analysis.")

    metrics_to_test = ['f1_1', 'accuracy']
    levels_to_process = [args.level] if args.level and args.level != 'all' else ALL_LEVELS

    for level in levels_to_process:
        results_dir_level = get_results_dir(level)
        # statistics_output_dir = results_dir_level / "_statistics"
        # statistics_output_dir.mkdir(parents=True, exist_ok=True)

        projects_to_run_tests_on = []
        if args.project != 'all':
            projects_to_run_tests_on.append(args.project)
        else:
            # If args.project is 'all' (default), find all project dirs at this level
            for item in results_dir_level.iterdir():
                if item.is_dir() and not item.name.startswith("_"): # Exclude _summary, _statistics
                    projects_to_run_tests_on.append(item.name)
            if not projects_to_run_tests_on:
                logging.warning(f"No project subdirectories found in {results_dir_level} for level '{level}' when args.project was 'all'. No Friedman tests will be run for this level.")
                continue
            logging.info(f"Running within-project Friedman tests for all detected projects at level '{level}': {projects_to_run_tests_on}")

        resampling_strategies_to_consider = []
        if args.resampling == 'all':
            resampling_strategies_to_consider = ['none'] + ALL_ACTUAL_RESAMPLING_METHODS
        elif args.resampling:
            resampling_strategies_to_consider = [args.resampling]

        for project_name_to_test in projects_to_run_tests_on:
            for resampling_strategy_name in resampling_strategies_to_consider:
                for current_metric in metrics_to_test:
                    logging.info(f"--- Friedman Test (Within-Project: '{project_name_to_test}') for Metric: '{current_metric}' (Level: {level}, Resampling: {resampling_strategy_name}) ---")

                    # --- Within-Project, Cross-Fold Friedman Test Logic ---
                    project_analysis_resampling_dir = results_dir_level / project_name_to_test / f"analysis_{resampling_strategy_name}"
                    statistics_output_dir = project_analysis_resampling_dir / "statistics"
                    statistics_output_dir.mkdir(parents=True, exist_ok=True)

                    if not project_analysis_resampling_dir.exists():
                        logging.warning(f"Analysis directory not found for project '{project_name_to_test}' at '{project_analysis_resampling_dir}'. Skipping Friedman test for this configuration.")
                        continue

                    fold_metric_files = list(project_analysis_resampling_dir.glob("*_fold_metrics.json"))
                    if not fold_metric_files:
                        logging.warning(f"No '*_fold_metrics.json' files found in '{project_analysis_resampling_dir}'. Skipping Friedman test.")
                        continue

                    model_performances_across_folds = []
                    algorithms_in_test = []
                    num_folds_expected = None

                    for f_path in fold_metric_files:
                        model_name_from_file = f_path.name.replace("_fold_metrics.json", "")
                        if model_name_from_file not in ALL_CLASSIFIER_FUNCTION_NAMES:
                            logging.debug(f"Skipping file {f_path.name} as it does not match known classifier naming for fold metrics.")
                            continue
                        try:
                            with open(f_path, 'r') as f_json:
                                single_model_fold_data = json.load(f_json)

                            metric_values_this_model = []
                            for fold_data_dict in single_model_fold_data:
                                if isinstance(fold_data_dict, dict) and current_metric in fold_data_dict and fold_data_dict[current_metric] is not None:
                                    metric_values_this_model.append(fold_data_dict[current_metric])
                                else:
                                    logging.warning(f"Metric '{current_metric}' missing or None in a fold for model '{model_name_from_file}' in {f_path}. This model will be excluded from this specific Friedman test.")
                                    metric_values_this_model = []
                                    break

                            if not metric_values_this_model:
                                continue

                            if num_folds_expected is None:
                                num_folds_expected = len(metric_values_this_model)
                            elif len(metric_values_this_model) != num_folds_expected:
                                logging.warning(f"Model '{model_name_from_file}' has {len(metric_values_this_model)} folds, expected {num_folds_expected} (from {f_path}). Excluding from this Friedman test.")
                                continue

                            model_performances_across_folds.append(metric_values_this_model)
                            algorithms_in_test.append(model_name_from_file)

                        except json.JSONDecodeError:
                            logging.error(f"Error decoding JSON from {f_path}. Skipping this file for Friedman test.")
                        except Exception as e:
                            logging.error(f"Error processing file {f_path} for model '{model_name_from_file}': {e}. Skipping this model.")

                    if len(model_performances_across_folds) < 2 or (num_folds_expected is not None and num_folds_expected < 2):
                        logging.warning(f"Not enough data for within-project Friedman test (Algorithms: {len(model_performances_across_folds)}, Folds: {num_folds_expected}). Needs at least 2 algorithms and 2 folds. Skipping for project '{project_name_to_test}', metric '{current_metric}'.")
                        continue

                    try:
                        stat, p_value = friedmanchisquare(*model_performances_across_folds)
                        logging.info(f"Within-project Friedman test for project '{project_name_to_test}', metric '{current_metric}' (Level: '{level}', Resampling: '{resampling_strategy_name}'):")
                        logging.info(f"  CV Folds (Blocks): {num_folds_expected}, Models (Groups): {len(algorithms_in_test)}")
                        logging.info(f"  Models Compared: {algorithms_in_test}")
                        logging.info(f"  Statistic: {stat:.4f}, P-value: {p_value:.4f}")

                        friedman_results = {
                            'test_type': 'within_project_cross_fold',
                            'project_tested': project_name_to_test,
                            'level': level,
                            'resampling_strategy': resampling_strategy_name,
                            'metric_tested': current_metric,
                            'num_cv_folds_ (blocks)': num_folds_expected,
                            'num_models_ (groups)': len(algorithms_in_test),
                            'models_compared': algorithms_in_test,
                            'friedman_statistic': stat,
                            'p_value': p_value,
                            'timestamp': datetime.datetime.now().isoformat()
                        }
                        results_file_name = f"friedman_test_{current_metric}.json"
                        results_file_path = statistics_output_dir / results_file_name
                        with open(results_file_path, 'w') as f_json_out:
                            json.dump(friedman_results, f_json_out, indent=4)
                        logging.info(f"Within-project Friedman test results saved to {results_file_path}")

                    except ValueError as ve:
                        logging.error(f"ValueError during within-project Friedman test for project '{project_name_to_test}', metric '{current_metric}': {ve}. (May occur if all models have identical performance in every fold)")
                    except Exception as e:
                        logging.error(f"Error performing within-project Friedman test for project '{project_name_to_test}', metric '{current_metric}': {e}")

    logging.info("Friedman test analysis finished.")

def run_nemenyi_analysis(args, alpha=0.05):
    """Performs Nemenyi post-hoc test if Friedman test was significant."""
    logging.info("Starting Nemenyi post-hoc test analysis.")

    metrics_to_test = ['f1_1', 'accuracy']
    levels_to_process = [args.level] if args.level and args.level != 'all' else ALL_LEVELS

    for level in levels_to_process:
        results_dir_level = get_results_dir(level)
        # statistics_output_dir = results_dir_level / "_statistics"
        # # Ensure statistics_output_dir exists (it should if Friedman ran)
        # statistics_output_dir.mkdir(parents=True, exist_ok=True)

        projects_to_consider = []
        if args.project != 'all':
            projects_to_consider.append(args.project)
        else:
            for item in results_dir_level.iterdir():
                if item.is_dir() and not item.name.startswith("_"):
                    projects_to_consider.append(item.name)
            if not projects_to_consider:
                logging.warning(f"No project subdirectories found in {results_dir_level} for Nemenyi tests (level '{level}').")
                continue
            logging.info(f"Nemenyi: Targeting all detected projects for level '{level}': {projects_to_consider}")

        resampling_strategies_to_consider = []
        if args.resampling == 'all':
            resampling_strategies_to_consider = ['none'] + ALL_ACTUAL_RESAMPLING_METHODS
        elif args.resampling:
            resampling_strategies_to_consider = [args.resampling]
        else:
            # If no resampling specified, default to 'none' for Nemenyi test
            resampling_strategies_to_consider = ['none']

        for project_name in projects_to_consider:
            for resampling_strategy_name in resampling_strategies_to_consider:
                for current_metric in metrics_to_test:
                    logging.info(f"--- Nemenyi Check for Project: '{project_name}', Metric: '{current_metric}', Level: {level}, Resampling: {resampling_strategy_name} ---")

                    project_resampling_dir = results_dir_level / project_name / f"analysis_{resampling_strategy_name}"
                    statistics_output_dir = project_resampling_dir / "statistics"
                    statistics_output_dir.mkdir(parents=True, exist_ok=True)

                    friedman_file_name = f"friedman_test_{current_metric}.json"
                    friedman_file_path = statistics_output_dir / friedman_file_name

                    if not friedman_file_path.exists():
                        logging.info(f"Friedman results file not found: {friedman_file_path}. Skipping Nemenyi test.")
                        continue

                    try:
                        with open(friedman_file_path, 'r') as f_friedman:
                            friedman_data = json.load(f_friedman)

                        if friedman_data.get('p_value') is None:
                            logging.warning(f"P-value missing in Friedman results {friedman_file_path}. Skipping Nemenyi.")
                            continue

                        if friedman_data['p_value'] >= alpha:
                            logging.info(f"Friedman test p-value ({friedman_data['p_value']:.4f}) is not < {alpha}. Nemenyi test not required for {friedman_file_name}.")
                            continue

                        logging.info(f"Friedman p-value ({friedman_data['p_value']:.4f}) < {alpha}. Proceeding with Nemenyi test for {friedman_file_name}.")

                        models_to_compare = friedman_data.get('models_compared')
                        expected_folds = friedman_data.get('num_cv_folds_ (blocks)')

                        if not models_to_compare or not expected_folds or len(models_to_compare) < 2 or expected_folds < 2:
                            logging.warning(f"Insufficient model or fold data in Friedman results ({friedman_file_name}) for Nemenyi. Models: {models_to_compare}, Folds: {expected_folds}. Skipping.")
                            continue

                        # Reload fold data for the specific models and metric
                        project_analysis_resampling_dir = results_dir_level / project_name / f"analysis_{resampling_strategy_name}"
                        model_fold_scores = {}

                        for model_name in models_to_compare:
                            fold_metric_file_path = project_analysis_resampling_dir / f"{model_name}_fold_metrics.json"
                            if not fold_metric_file_path.exists():
                                logging.error(f"CRITICAL: Fold metric file {fold_metric_file_path} (listed in significant Friedman test) not found. Cannot run Nemenyi. Skipping.")
                                model_fold_scores = {} # Invalidate data for this Nemenyi test
                                break

                            current_model_scores = []
                            try:
                                with open(fold_metric_file_path, 'r') as f_fold_json:
                                    single_model_all_fold_data = json.load(f_fold_json)
                                for fold_data_item in single_model_all_fold_data:
                                    if isinstance(fold_data_item, dict) and current_metric in fold_data_item and fold_data_item[current_metric] is not None:
                                        current_model_scores.append(fold_data_item[current_metric])
                                    else:
                                        logging.error(f"CRITICAL: Metric '{current_metric}' missing or None in a fold for model '{model_name}' from {fold_metric_file_path}, which was part of a significant Friedman test. Invalidating Nemenyi for this config.")
                                        current_model_scores = [] # Discard
                                        break

                                if not current_model_scores or len(current_model_scores) != expected_folds:
                                    logging.error(f"CRITICAL: Data inconsistency for Nemenyi. Model '{model_name}' from {fold_metric_file_path} had {len(current_model_scores)} valid scores for metric '{current_metric}', but Friedman test indicated {expected_folds} folds. Invalidating Nemenyi.")
                                    model_fold_scores = {} # Invalidate
                                    break
                                model_fold_scores[model_name] = current_model_scores
                            except Exception as e_load_fold:
                                logging.error(f"CRITICAL: Error loading or processing fold metric file {fold_metric_file_path} for Nemenyi: {e_load_fold}. Invalidating Nemenyi for this config.")
                                model_fold_scores = {} # Invalidate
                                break

                        if not model_fold_scores or len(model_fold_scores) < 2: # Check if data got invalidated
                            logging.error(f"Nemenyi test aborted for {friedman_file_name} due to critical errors in loading or validating fold data consistent with Friedman results.")
                            continue

                        # Prepare DataFrame for Nemenyi test
                        df_for_nemenyi = pd.DataFrame(model_fold_scores)

                        # Perform Nemenyi test
                        nemenyi_results_df = sp.posthoc_nemenyi_friedman(df_for_nemenyi)

                        nemenyi_output_filename = f"nemenyi_test_{current_metric}.csv"
                        nemenyi_output_path = statistics_output_dir / nemenyi_output_filename
                        nemenyi_results_df.to_csv(nemenyi_output_path)
                        logging.info(f"Nemenyi post-hoc test results saved to {nemenyi_output_path}")

                        # Generate and save the significance heatmap
                        heatmap_filename = f"significance_heatmap_{current_metric}.png"
                        heatmap_path = statistics_output_dir / heatmap_filename
                        heatmap_title = (
                            f"Pairwise Significance for '{current_metric}'\n"
                            f"(Project: {project_name}, Level: {level}, Resampling: {resampling_strategy_name})"
                        )
                        plot_significance_heatmap(p_values_df=nemenyi_results_df, title=heatmap_title, output_path=heatmap_path)

                        # Generate and save CD diagram using scikit-posthocs
                        try:
                            logging.info("Generating CD diagram with scikit-posthocs.")
                            # Calculate mean ranks for the CD diagram. Higher score -> lower rank value (better).
                            mean_ranks = df_for_nemenyi.rank(axis=1, ascending=False).mean()

                            fig, ax = plt.subplots(figsize=(10, max(4, 0.5 * len(df_for_nemenyi.columns))), dpi=200)

                            # Call with mean ranks and the significance matrix (p-values) from Nemenyi test
                            sp.critical_difference_diagram(mean_ranks, nemenyi_results_df, ax=ax)

                            ax.set_title(f"Critical Difference Diagram (scikit-posthocs) for '{current_metric}'\n"
                                        f"(Project: {project_name}, Level: {level}, Resampling: {resampling_strategy_name})")

                            sp_cd_diagram_filename = f"scikit_cd_diagram_{current_metric}.png"
                            sp_cd_diagram_path = statistics_output_dir / sp_cd_diagram_filename
                            plt.savefig(sp_cd_diagram_path, bbox_inches='tight')
                            plt.close(fig)
                            logging.info(f"Scikit-posthocs CD diagram saved to {sp_cd_diagram_path}")
                        except AttributeError:
                            logging.warning("Function 'critical_difference_diagram' not found in scikit-posthocs. You may need to install a different version or an extension. Skipping this plot.")
                        except Exception as e_sp_cd:
                            logging.error(f"Failed to generate scikit-posthocs CD diagram: {e_sp_cd}", exc_info=True)
                            plt.close('all') # Close any dangling plots on error

                        # Generate and save CD diagram using autorank
                        try:
                            logging.info("Generating CD diagram with autorank.")
                            autorank_result = autorank.autorank(df_for_nemenyi, alpha=alpha, verbose=False)
                            autorank.plot_stats(autorank_result)
                            autorank_cd_diagram_filename = f"autorank_cd_diagram_{current_metric}.png"
                            autorank_cd_diagram_path = statistics_output_dir / autorank_cd_diagram_filename
                            plt.savefig(autorank_cd_diagram_path, bbox_inches='tight')
                            plt.close()
                            logging.info(f"Autorank CD diagram saved to {autorank_cd_diagram_path}")
                        except Exception as e_autorank:
                            logging.error(f"Failed to generate autorank CD diagram: {e_autorank}")

                    except json.JSONDecodeError:
                        logging.error(f"Error decoding Friedman JSON from {friedman_file_path}. Skipping Nemenyi.")
                    except Exception as e_outer:
                        logging.error(f"Error during Nemenyi processing for {friedman_file_path}: {e_outer}")

    logging.info("Nemenyi post-hoc test analysis finished.")

def run_wilcoxon_go_metrics_analysis(args):
    """
    Performs Wilcoxon signed-rank test to compare results with and without Go-specific metrics.
    This paired test determines if excluding Go metrics significantly affects model performance.

    Uses the new hierarchical directory structure:
        results_{level}_level/{project}/{cv_type}/{feature_set}/{resampling}/

    Compares:
        - {project}/{cv_type}/full/{resampling}/ (WITH Go-metrics)
        - {project}/{cv_type}/no_go_metrics/{resampling}/ (WITHOUT Go-metrics)
    """
    print("\n" + "="*70)
    print("Wilcoxon Signed-Rank Test: Go-Metrics Impact Analysis")
    print("="*70)
    logging.info("Starting Wilcoxon signed-rank test for Go-metrics impact analysis.")

    if not args.level:
        logging.error("--level is required for Wilcoxon Go-metrics test.")
        print("ERROR: --level is required for Wilcoxon Go-metrics test.")
        return

    levels_to_process = [args.level] if args.level != 'all' else ALL_LEVELS
    metrics_to_compare = ['f1_1', 'accuracy', 'precision_1', 'recall_1', 'auc', 'mcc', 'pr_auc']

    # Determine CV type from args
    cv_type = get_cv_type_name(getattr(args, 'shuffle_cv', False))

    for level in levels_to_process:
        print(f"\nAnalyzing Level: {level.upper()}")
        logging.info(f"===== Wilcoxon Test for Go-Metrics Impact at Level: {level} =====")

        results_dir_base = get_results_dir(level)

        print(f"✓ Using hierarchical directory structure")
        print(f"  CV Type: {cv_type}")
        print(f"  WITH Go-metrics: .../{cv_type}/full/")
        print(f"  WITHOUT Go-metrics: .../{cv_type}/no_go_metrics/")
        logging.info(f"Using hierarchical structure with cv_type={cv_type}")

        # Determine resampling strategies to analyze
        resampling_strategies = []
        if args.resampling == 'all':
            resampling_strategies = ['none'] + ALL_ACTUAL_RESAMPLING_METHODS
        else:
            resampling_strategies = [args.resampling]

        for resampling_strategy in resampling_strategies:
            print(f"\nResampling Strategy: {resampling_strategy}")
            logging.info(f"--- Analyzing Resampling Strategy: {resampling_strategy} ---")

            # Get list of projects to analyze using hierarchical structure
            projects_to_analyze = []
            if args.project == 'all':
                # Find all projects that have results
                for item in results_dir_base.iterdir():
                    if item.is_dir() and not item.name.startswith("_"):
                        projects_to_analyze.append(item.name)
            else:
                projects_to_analyze = [args.project]

            if not projects_to_analyze:
                logging.warning(f"No projects found for level {level}")
                print(f"  WARNING: No projects found")
                continue

            print(f"  Found {len(projects_to_analyze)} projects to analyze")

            # For each metric, collect paired samples
            for metric in metrics_to_compare:
                logging.info(f"Testing metric: {metric}")

                paired_data_with_go = []
                paired_data_without_go = []

                for project_name in projects_to_analyze:
                    # New hierarchical structure:
                    # {project}/{cv_type}/full/{resampling}/ vs {project}/{cv_type}/no_go_metrics/{resampling}/
                    analysis_with_go = results_dir_base / project_name / cv_type / 'full' / resampling_strategy
                    analysis_without_go = results_dir_base / project_name / cv_type / 'no_go_metrics' / resampling_strategy

                    if not analysis_with_go.exists():
                        logging.debug(f"Analysis directory with Go-metrics not found for {project_name}: {analysis_with_go}")
                        continue

                    if not analysis_without_go.exists():
                        logging.debug(f"Analysis directory without Go-metrics not found for {project_name}: {analysis_without_go}")
                        continue

                    # For each model, collect metrics from both runs
                    for model_name in ALL_CLASSIFIER_FUNCTION_NAMES:
                        fold_metrics_with = analysis_with_go / f"{model_name}_fold_metrics.json"
                        fold_metrics_without = analysis_without_go / f"{model_name}_fold_metrics.json"

                        if not fold_metrics_with.exists() or not fold_metrics_without.exists():
                            continue

                        try:
                            with open(fold_metrics_with, 'r') as f:
                                metrics_with = json.load(f)
                            with open(fold_metrics_without, 'r') as f:
                                metrics_without = json.load(f)

                            # Extract the metric values from each fold
                            for fold_with, fold_without in zip(metrics_with, metrics_without):
                                if metric in fold_with and metric in fold_without:
                                    if fold_with[metric] is not None and fold_without[metric] is not None:
                                        paired_data_with_go.append(fold_with[metric])
                                        paired_data_without_go.append(fold_without[metric])

                        except Exception as e:
                            logging.error(f"Error loading metrics for {project_name}/{model_name}: {e}")
                            continue

                # Perform Wilcoxon signed-rank test
                if len(paired_data_with_go) < 5:  # Need at least 5 pairs for reliable test
                    logging.warning(f"Insufficient paired samples ({len(paired_data_with_go)}) for {metric} at {level}/{resampling_strategy}. Need at least 5.")
                    continue

                try:
                    # Wilcoxon signed-rank test
                    statistic, p_value = wilcoxon(paired_data_with_go, paired_data_without_go,
                                                   alternative='two-sided', zero_method='wilcox')

                    # Calculate effect size (mean difference and median difference)
                    differences = np.array(paired_data_with_go) - np.array(paired_data_without_go)
                    mean_diff = np.mean(differences)
                    median_diff = np.median(differences)

                    mean_with = np.mean(paired_data_with_go)
                    mean_without = np.mean(paired_data_without_go)

                    logging.info(f"Wilcoxon Test Results for {metric} (Level: {level}, Resampling: {resampling_strategy}):")
                    logging.info(f"  Number of paired samples: {len(paired_data_with_go)}")
                    logging.info(f"  Mean WITH Go-metrics: {mean_with:.4f}")
                    logging.info(f"  Mean WITHOUT Go-metrics: {mean_without:.4f}")
                    logging.info(f"  Mean difference (WITH - WITHOUT): {mean_diff:.4f}")
                    logging.info(f"  Median difference: {median_diff:.4f}")
                    logging.info(f"  Wilcoxon statistic: {statistic:.4f}")
                    logging.info(f"  P-value: {p_value:.4f}")

                    significance = "SIGNIFICANT" if p_value < 0.05 else "NOT significant"
                    direction = "FAVORS WITH Go-metrics" if mean_diff > 0 else "FAVORS WITHOUT Go-metrics" if mean_diff < 0 else "NO clear direction"
                    logging.info(f"  Result: {significance} (α=0.05) - {direction}")

                    # Save results
                    wilcoxon_results = {
                        'test_type': 'wilcoxon_signed_rank',
                        'level': level,
                        'resampling_strategy': resampling_strategy,
                        'metric_tested': metric,
                        'n_pairs': len(paired_data_with_go),
                        'mean_with_go_metrics': float(mean_with),
                        'mean_without_go_metrics': float(mean_without),
                        'mean_difference': float(mean_diff),
                        'median_difference': float(median_diff),
                        'wilcoxon_statistic': float(statistic),
                        'p_value': float(p_value),
                        'significant_at_0.05': p_value < 0.05,
                        'direction': direction,
                        'cv_type': cv_type,
                        'timestamp': datetime.datetime.now().isoformat()
                    }

                    # Create statistics directory for this comparison
                    # Structure: _statistics/{cv_type}/go_metrics_comparison/
                    stats_dir = results_dir_base / "_statistics" / cv_type / "go_metrics_comparison"
                    stats_dir.mkdir(parents=True, exist_ok=True)

                    results_file = stats_dir / f"wilcoxon_{level}_{resampling_strategy}_{metric}.json"
                    with open(results_file, 'w') as f:
                        # Use convert_numpy_to_list_recursive to handle numpy types
                        json.dump(convert_numpy_to_list_recursive(wilcoxon_results), f, indent=4)
                    logging.info(f"Results saved to {results_file}")
                    print(f"✓ {metric}: p={p_value:.4f} | Diff={mean_diff:+.4f} | {direction}")

                except ValueError as ve:
                    logging.error(f"ValueError in Wilcoxon test for {metric}: {ve}")
                except Exception as e:
                    logging.error(f"Error performing Wilcoxon test for {metric}: {e}")

    # Generate summary report
    print("\n" + "-"*70)
    print("Generating summary report...")
    generate_wilcoxon_summary_report(args)
    logging.info("Wilcoxon signed-rank test for Go-metrics impact finished.")
    print("="*70)
    print("Wilcoxon test completed!")
    print("="*70 + "\n")

def generate_wilcoxon_summary_report(args):
    """Generate a markdown summary of all Wilcoxon test results."""
    levels_to_process = [args.level] if args.level != 'all' else ALL_LEVELS
    cv_type = get_cv_type_name(getattr(args, 'shuffle_cv', False))

    for level in levels_to_process:
        results_dir_base = get_results_dir(level)
        stats_dir = results_dir_base / "_statistics" / cv_type / "go_metrics_comparison"

        if not stats_dir.exists():
            print(f"  Note: Stats directory not found: {stats_dir}")
            continue

        # Collect all result files
        result_files = list(stats_dir.glob("wilcoxon_*.json"))
        if not result_files:
            print(f"  Note: No result files found in {stats_dir}")
            continue

        print(f"  Found {len(result_files)} result files")

        all_results = []
        for file_path in result_files:
            try:
                with open(file_path, 'r') as f:
                    all_results.append(json.load(f))
            except Exception as e:
                logging.error(f"Error reading {file_path}: {e}")
                print(f"  Error reading {file_path.name}: {e}")

        if not all_results:
            print(f"  Note: No valid results could be loaded")
            continue

        print(f"  Loaded {len(all_results)} results")

        # Create markdown report
        markdown_content = f"""# Wilcoxon Signed-Rank Test: Impact of Go-Specific Metrics

## Level: {level.title()}

This report presents the results of paired Wilcoxon signed-rank tests comparing model performance
WITH and WITHOUT Go-specific metrics. The test determines whether excluding Go-specific features
significantly affects predictive performance.

### Go-Specific Metrics Excluded:
- struct_count, interface_count, goroutine_count, channel_count
- defer_count, context_usage_count, json_tag_count
- variadic_function_count, pointer_receiver_count, error_handling_count

"""

        # Group by resampling strategy
        by_resampling = {}
        for result in all_results:
            resampling = result['resampling_strategy']
            if resampling not in by_resampling:
                by_resampling[resampling] = []
            by_resampling[resampling].append(result)

        for resampling, results in sorted(by_resampling.items()):
            markdown_content += f"\n## Resampling Strategy: {resampling}\n\n"

            table_data = []
            headers = ["Metric", "N Pairs", "Mean WITH", "Mean WITHOUT", "Difference", "P-value", "Significant?", "Direction"]

            for result in sorted(results, key=lambda x: x['metric_tested']):
                sig_marker = "✓" if result['significant_at_0.05'] else "✗"
                table_data.append([
                    result['metric_tested'],
                    result['n_pairs'],
                    f"{result['mean_with_go_metrics']:.4f}",
                    f"{result['mean_without_go_metrics']:.4f}",
                    f"{result['mean_difference']:.4f}",
                    f"{result['p_value']:.4f}",
                    sig_marker,
                    result['direction'].replace("FAVORS ", "")
                ])

            markdown_content += tabulate.tabulate(table_data, headers=headers, tablefmt='pipe')
            markdown_content += "\n\n"

        markdown_content += f"""
### Interpretation:
- **Significant?**: ✓ indicates p < 0.05 (statistically significant difference)
- **Difference**: Positive values indicate better performance WITH Go-metrics
- **Direction**: Shows which configuration performs better on average

*Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""

        # Save report
        report_path = stats_dir / f"wilcoxon_summary_{level}.md"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(markdown_content)
        logging.info(f"Wilcoxon summary report saved to {report_path}")
        print(f"\n✓ Summary report saved to:")
        print(f"  {report_path}")
        print(f"\n✓ Individual results saved to:")
        print(f"  {stats_dir}/")

def parse_markdown_table(markdown_content, table_title):
    """
    Parses a markdown table with a given title from a string.
    Returns headers and rows as lists.
    """
    lines = markdown_content.split('\n')
    table_lines = []
    in_table_section = False

    # Find the table title section
    for i, line in enumerate(lines):
        # Match "## Title" or "### Title"
        if line.strip().startswith('#') and table_title in line:
            in_table_section = True
            # Search for the table starting from the next line
            for j in range(i + 1, len(lines)):
                current_line = lines[j].strip()
                if current_line.startswith('|'):
                    table_lines.append(current_line)
                elif table_lines: # If we have already found table lines and this one doesn't start with '|', table has ended
                    break
            if table_lines: # Found the table for the given title
                break

    if not table_lines:
        logging.warning(f"Could not find a markdown table under title '{table_title}'")
        return None, None

    # First line is headers
    headers = [h.strip() for h in table_lines[0].strip('|').split('|')]
    # Second line is separator, e.g. |---|---|
    # Third line onwards is data
    rows = []
    for row_line in table_lines[2:]:
        row_line = row_line.strip()
        if not row_line.startswith('|'):
            # This handles cases where there are blank lines after the table.
            break
        row = [r.strip() for r in row_line.strip('|').split('|')]
        if len(row) == len(headers):
            rows.append(row)
        else:
            logging.warning(f"Skipping malformed row in table '{table_title}': '{row_line}' (expected {len(headers)} cells, got {len(row)})")

    return headers, rows

def plot_best_f1_per_project(level, results_dir):
    """
    Generates a bar chart of the best F1-score for each project at a given level.
    """
    results_md_path = results_dir / "results.md"
    if not results_md_path.exists():
        logging.warning(f"Top-level results.md not found at {results_md_path}. Skipping plot.")
        return

    with open(results_md_path, 'r', encoding='utf-8') as f:
        content = f.read()

    headers, rows = parse_markdown_table(content, "Best F1-Score Result for Each Project")

    if not headers or not rows:
        logging.warning(f"Could not parse 'Best F1-Score Result for Each Project' table from {results_md_path}. Skipping plot.")
        return

    try:
        df = pd.DataFrame(rows, columns=headers)
        # Find the columns for Project and F1-Score. The name might vary.
        project_col = 'Project'
        f1_col = 'F1-Score (Bug)'
        if project_col not in df.columns or f1_col not in df.columns:
            logging.error(f"Required columns '{project_col}' or '{f1_col}' not in table. Available: {df.columns}")
            return

        df[f1_col] = pd.to_numeric(df[f1_col], errors='coerce')
        df = df.dropna(subset=[f1_col])
        df = df.sort_values(by=f1_col, ascending=False)
    except Exception as e:
        logging.error(f"Error processing parsed table data for {level} best F1 plot: {e}")
        return

    if df.empty:
        logging.warning(f"No data to plot for best F1-scores for level {level}.")
        return

    plt.figure(figsize=(max(10, len(df) * 0.6), 7))
    colors = plt.cm.plasma(np.linspace(0.2, 0.8, len(df)))
    bars = plt.bar(df[project_col], df[f1_col], color=colors)

    plt.xlabel('Project', fontsize=12)
    plt.ylabel('Best F1-Score (Bug Class)', fontsize=12)
    plt.title(f'Best F1-Score per Project - {level.title()} Level', fontsize=14)
    plt.xticks(rotation=45, ha="right", fontsize=10)
    plt.yticks(np.arange(0, 1.1, 0.1), fontsize=10)
    plt.ylim(0, 1.05)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.015, f'{yval:.3f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout(pad=1.5)

    plot_path = results_dir / f"best_f1_scores_per_project_{level}_level.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    logging.info(f"Best F1-scores per project plot saved to {plot_path}")


def plot_f1_scores_per_resampling(level, results_dir):
    """
    For each project, generates a bar chart of F1 scores per resampling method.
    """
    project_dirs = [d for d in results_dir.iterdir() if d.is_dir() and not d.name.startswith('_')]

    for project_dir in project_dirs:
        results_md_path = project_dir / "results.md"
        if not results_md_path.exists():
            continue

        with open(results_md_path, 'r', encoding='utf-8') as f:
            content = f.read()

        headers, rows = parse_markdown_table(content, "Results by Resampling Method")

        if not headers or not rows:
            logging.warning(f"Could not parse 'Results by Resampling Method' table for {project_dir.name}. Skipping plot.")
            continue

        try:
            df = pd.DataFrame(rows, columns=headers)
            resampling_col = 'Resampling Method'
            f1_col = 'F1-Score (Bug)'
            if resampling_col not in df.columns or f1_col not in df.columns:
                logging.error(f"Required columns '{resampling_col}' or '{f1_col}' not in table for {project_dir.name}. Available: {df.columns}")
                continue

            df[f1_col] = pd.to_numeric(df[f1_col], errors='coerce')
            df = df.dropna(subset=[f1_col])
            df = df.sort_values(by=f1_col, ascending=False)
        except Exception as e:
            logging.error(f"Error processing parsed table data for {project_dir.name} F1 plot: {e}")
            continue

        if df.empty:
            logging.warning(f"No data to plot for F1-scores by resampling for project {project_dir.name}.")
            continue

        plt.figure(figsize=(max(10, len(df) * 0.7), 7))
        colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(df)))
        bars = plt.bar(df[resampling_col], df[f1_col], color=colors)

        plt.xlabel('Resampling Method', fontsize=12)
        plt.ylabel('Best F1-Score (Bug Class)', fontsize=12)
        plt.title(f'F1-Scores by Resampling Method - {project_dir.name} ({level.title()} Level)', fontsize=14)
        plt.xticks(rotation=45, ha="right", fontsize=10)
        plt.yticks(np.arange(0, 1.1, 0.1), fontsize=10)
        plt.ylim(0, 1.05)
        plt.grid(axis='y', linestyle='--', alpha=0.7)

        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.015, f'{yval:.3f}', ha='center', va='bottom', fontsize=9)

        plt.tight_layout(pad=1.5)

        plot_path = project_dir / f"f1_scores_by_resampling.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        logging.info(f"F1-scores by resampling plot saved to {plot_path}")

def plot_f1_scores_by_ml_algorithm(level, results_dir):
    """
    For each project, generates a bar chart of F1 scores per ML algorithm.
    """
    project_dirs = [d for d in results_dir.iterdir() if d.is_dir() and not d.name.startswith('_')]

    for project_dir in project_dirs:
        results_md_path = project_dir / "results.md"
        if not results_md_path.exists():
            continue

        with open(results_md_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # The table is under "Algorithm Performance Summary"
        headers, rows = parse_markdown_table(content, "Algorithm Performance Summary")

        if not headers or not rows:
            logging.warning(f"Could not parse 'Algorithm Performance Summary' table for {project_dir.name}. Skipping plot.")
            continue

        try:
            df = pd.DataFrame(rows, columns=headers)
            ml_col = 'ML Algorithm'
            f1_col = 'F1-Score (Bug)'
            if ml_col not in df.columns or f1_col not in df.columns:
                logging.error(f"Required columns '{ml_col}' or '{f1_col}' not in table for {project_dir.name}. Available: {df.columns}")
                continue

            df[f1_col] = pd.to_numeric(df[f1_col], errors='coerce')
            df = df.dropna(subset=[f1_col])
            df = df.sort_values(by=f1_col, ascending=False)
        except Exception as e:
            logging.error(f"Error processing parsed table data for {project_dir.name} ML algorithm F1 plot: {e}")
            continue

        if df.empty:
            logging.warning(f"No data to plot for F1-scores by ML algorithm for project {project_dir.name}.")
            continue

        plt.figure(figsize=(max(10, len(df) * 0.7), 7))
        colors = plt.cm.cividis(np.linspace(0.2, 0.8, len(df))) # Different color scheme
        bars = plt.bar(df[ml_col], df[f1_col], color=colors)

        plt.xlabel('ML Algorithm', fontsize=12)
        plt.ylabel('Best F1-Score (Bug Class)', fontsize=12)
        plt.title(f'F1-Scores by ML Algorithm - {project_dir.name} ({level.title()} Level)', fontsize=14)
        plt.xticks(rotation=45, ha="right", fontsize=10)
        plt.yticks(np.arange(0, 1.1, 0.1), fontsize=10)
        plt.ylim(0, 1.05)
        plt.grid(axis='y', linestyle='--', alpha=0.7)

        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.015, f'{yval:.3f}', ha='center', va='bottom', fontsize=9)

        plt.tight_layout(pad=1.5)

        plot_path = project_dir / f"f1_scores_by_ml_algorithm.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        logging.info(f"F1-scores by ML algorithm plot saved to {plot_path}")


# =============================================================================
# Comprehensive Report Generation (--generate-reports)
# =============================================================================

def generate_comprehensive_reports(level, cv_type='temporal', feature_set='full'):
    """
    Generate comprehensive reports for a specific level and CV type.

    Creates:
    - results.md with CV metrics (mean ± std) and holdout metrics
    - f1_scores_by_ml_algorithm.png
    - f1_scores_by_resampling.png
    - holdout_comparison.png

    Output structure:
        results_{level}_level/{project}/{cv_type}/{feature_set}/results.md
        results_{level}_level/{project}/{cv_type}/{feature_set}/f1_scores_by_ml_algorithm.png
        results_{level}_level/{project}/{cv_type}/{feature_set}/f1_scores_by_resampling.png
        results_{level}_level/{project}/{cv_type}/{feature_set}/cv_vs_holdout_mcc.png
    """
    results_dir = get_results_dir(level)

    # Find all projects
    project_dirs = [d for d in results_dir.iterdir()
                    if d.is_dir() and not d.name.startswith('_')
                    and d.name not in ['combined', 'combine']]

    if not project_dirs:
        logging.warning(f"No project directories found in {results_dir}")
        return

    for project_dir in project_dirs:
        project_name = project_dir.name
        feature_set_dir = project_dir / cv_type / feature_set

        if not feature_set_dir.exists():
            logging.warning(f"Feature set directory not found: {feature_set_dir}")
            continue

        logging.info(f"Generating reports for {project_name}/{cv_type}/{feature_set}...")

        # Collect results from all resampling strategies
        all_results = collect_project_results_from_feature_set(feature_set_dir)

        if not all_results:
            logging.warning(f"No results found for {project_name}/{cv_type}/{feature_set}")
            continue

        # Generate results.md in feature_set directory
        generate_project_report_md(all_results, project_name, level, cv_type, feature_set, feature_set_dir)

        # Generate plots in feature_set directory
        generate_project_plots(all_results, project_name, level, cv_type, feature_set, feature_set_dir)

    # Generate top-level summary
    generate_level_summary_report(level, cv_type, feature_set, results_dir)


def collect_project_results_from_feature_set(feature_set_dir):
    """
    Collect all results from analysis_summary.json files in a feature_set directory.

    Returns list of dicts with metrics for each resampling strategy.
    """
    results = []

    if not feature_set_dir.exists():
        return results

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

                results.append({
                    'resampling': resampling,
                    'model': model_name,
                    # CV metrics with std
                    'cv_accuracy': cv_metrics.get('accuracy'),
                    'cv_accuracy_std': cv_metrics.get('accuracy_std'),
                    'cv_f1_bug': cv_metrics.get('f1_bug'),
                    'cv_f1_bug_std': cv_metrics.get('f1_bug_std'),
                    'cv_precision_bug': cv_metrics.get('precision_bug'),
                    'cv_precision_bug_std': cv_metrics.get('precision_bug_std'),
                    'cv_recall_bug': cv_metrics.get('recall_bug'),
                    'cv_recall_bug_std': cv_metrics.get('recall_bug_std'),
                    'cv_roc_auc': cv_metrics.get('roc_auc'),
                    'cv_roc_auc_std': cv_metrics.get('roc_auc_std'),
                    'cv_pr_auc': cv_metrics.get('pr_auc'),
                    'cv_pr_auc_std': cv_metrics.get('pr_auc_std'),
                    'cv_mcc': cv_metrics.get('mcc'),
                    'cv_mcc_std': cv_metrics.get('mcc_std'),
                    # Holdout metrics
                    'holdout_accuracy': holdout_metrics.get('accuracy'),
                    'holdout_f1_bug': holdout_metrics.get('f1_bug'),
                    'holdout_precision_bug': holdout_metrics.get('precision_bug'),
                    'holdout_recall_bug': holdout_metrics.get('recall_bug'),
                    'holdout_roc_auc': holdout_metrics.get('roc_auc'),
                    'holdout_pr_auc': holdout_metrics.get('pr_auc'),
                    'holdout_mcc': holdout_metrics.get('mcc'),
                })

        except Exception as e:
            logging.error(f"Error reading {summary_file}: {e}")

    return results


def collect_project_results(project_dir, cv_type, feature_set):
    """
    Collect all results from analysis_summary.json files for a project.
    DEPRECATED: Use collect_project_results_from_feature_set instead.
    """
    feature_set_dir = project_dir / cv_type / feature_set
    return collect_project_results_from_feature_set(feature_set_dir)


def format_metric_with_std(mean, std, decimals=4):
    """Format metric as 'mean ± std' or just 'mean' if std is None."""
    if mean is None:
        return 'N/A'
    if std is not None and not np.isnan(std):
        return f"{mean:.{decimals}f} ± {std:.{decimals}f}"
    return f"{mean:.{decimals}f}"


def generate_project_report_md(results, project_name, level, cv_type, feature_set, output_dir):
    """Generate comprehensive results.md for a project with CV and holdout metrics."""

    if not results:
        return

    df = pd.DataFrame(results)

    # Find best result by holdout MCC (or F1 if MCC not available)
    metric_col = 'holdout_mcc' if 'holdout_mcc' in df.columns and df['holdout_mcc'].notna().any() else 'holdout_f1_bug'
    if df[metric_col].notna().any():
        best_idx = df[metric_col].idxmax()
        best_result = df.loc[best_idx]
    else:
        best_result = df.iloc[0] if len(df) > 0 else None

    # Prepare markdown content
    cv_type_display = "Temporal CV" if cv_type == "temporal" else "Shuffle CV"
    feature_set_display = "Full Features" if feature_set == "full" else ("Without Go Metrics" if feature_set == "no_go_metrics" else f"Feature Set: {feature_set}")

    md_content = f"""# {project_name} - {level.title()} Level Analysis ({cv_type_display}, {feature_set_display})

## Best Overall Performance

| Metric | CV (Mean ± Std) | Holdout |
|--------|-----------------|---------|
| Resampling | {best_result['resampling']} | - |
| ML Algorithm | {best_result['model']} | - |
| Accuracy | {format_metric_with_std(best_result['cv_accuracy'], best_result['cv_accuracy_std'])} | {format_metric(best_result['holdout_accuracy'])} |
| F1-Score (Bug) | {format_metric_with_std(best_result['cv_f1_bug'], best_result['cv_f1_bug_std'])} | {format_metric(best_result['holdout_f1_bug'])} |
| Precision (Bug) | {format_metric_with_std(best_result['cv_precision_bug'], best_result['cv_precision_bug_std'])} | {format_metric(best_result['holdout_precision_bug'])} |
| Recall (Bug) | {format_metric_with_std(best_result['cv_recall_bug'], best_result['cv_recall_bug_std'])} | {format_metric(best_result['holdout_recall_bug'])} |
| ROC-AUC | {format_metric_with_std(best_result['cv_roc_auc'], best_result['cv_roc_auc_std'])} | {format_metric(best_result['holdout_roc_auc'])} |
| PR-AUC | {format_metric_with_std(best_result['cv_pr_auc'], best_result['cv_pr_auc_std'])} | {format_metric(best_result['holdout_pr_auc'])} |
| MCC | {format_metric_with_std(best_result['cv_mcc'], best_result['cv_mcc_std'])} | {format_metric(best_result['holdout_mcc'])} |

*Best result selected based on highest Holdout MCC*

"""

    # Results by Resampling Method (best model for each)
    md_content += "## Results by Resampling Method\n\n"
    md_content += "| Resampling | Best Model | CV F1 (Mean ± Std) | Holdout F1 | Holdout MCC |\n"
    md_content += "|------------|------------|-------------------|------------|-------------|\n"

    for resampling, group in df.groupby('resampling'):
        if group['holdout_mcc'].notna().any():
            best_for_resampling = group.loc[group['holdout_mcc'].idxmax()]
        elif group['holdout_f1_bug'].notna().any():
            best_for_resampling = group.loc[group['holdout_f1_bug'].idxmax()]
        else:
            best_for_resampling = group.iloc[0]

        md_content += f"| {resampling} | {best_for_resampling['model']} | "
        md_content += f"{format_metric_with_std(best_for_resampling['cv_f1_bug'], best_for_resampling['cv_f1_bug_std'])} | "
        md_content += f"{format_metric(best_for_resampling['holdout_f1_bug'])} | "
        md_content += f"{format_metric(best_for_resampling['holdout_mcc'])} |\n"

    # Algorithm Performance Summary
    md_content += "\n## Algorithm Performance Summary\n\n"
    md_content += "| ML Algorithm | Best Resampling | CV F1 (Mean ± Std) | Holdout F1 | Holdout MCC |\n"
    md_content += "|--------------|-----------------|-------------------|------------|-------------|\n"

    for model, group in df.groupby('model'):
        if group['holdout_mcc'].notna().any():
            best_for_model = group.loc[group['holdout_mcc'].idxmax()]
        elif group['holdout_f1_bug'].notna().any():
            best_for_model = group.loc[group['holdout_f1_bug'].idxmax()]
        else:
            best_for_model = group.iloc[0]

        md_content += f"| {model} | {best_for_model['resampling']} | "
        md_content += f"{format_metric_with_std(best_for_model['cv_f1_bug'], best_for_model['cv_f1_bug_std'])} | "
        md_content += f"{format_metric(best_for_model['holdout_f1_bug'])} | "
        md_content += f"{format_metric(best_for_model['holdout_mcc'])} |\n"

    # Detailed holdout results
    md_content += "\n## Detailed Holdout Test Results\n\n"
    md_content += "| Resampling | Model | Accuracy | F1 (Bug) | Precision | Recall | ROC-AUC | PR-AUC | MCC |\n"
    md_content += "|------------|-------|----------|----------|-----------|--------|---------|--------|-----|\n"

    # Sort by holdout MCC
    df_sorted = df.sort_values('holdout_mcc', ascending=False, na_position='last')

    for _, row in df_sorted.iterrows():
        md_content += f"| {row['resampling']} | {row['model']} | "
        md_content += f"{format_metric(row['holdout_accuracy'])} | "
        md_content += f"{format_metric(row['holdout_f1_bug'])} | "
        md_content += f"{format_metric(row['holdout_precision_bug'])} | "
        md_content += f"{format_metric(row['holdout_recall_bug'])} | "
        md_content += f"{format_metric(row['holdout_roc_auc'])} | "
        md_content += f"{format_metric(row['holdout_pr_auc'])} | "
        md_content += f"{format_metric(row['holdout_mcc'])} |\n"

    md_content += f"\n*Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n"

    # Save
    output_file = output_dir / "results.md"
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(md_content)
        logging.info(f"Report saved to {output_file}")
    except Exception as e:
        logging.error(f"Error saving report: {e}")


def generate_project_plots(results, project_name, level, cv_type, feature_set, output_dir):
    """Generate plots for a project's results."""

    if not results:
        return

    df = pd.DataFrame(results)
    cv_type_display = "Temporal CV" if cv_type == "temporal" else "Shuffle CV"
    feature_set_display = "Full Features" if feature_set == "full" else ("Without Go Metrics" if feature_set == "no_go_metrics" else f"FS: {feature_set}")

    # 1. F1 Scores by ML Algorithm (Holdout)
    try:
        model_best = df.groupby('model').apply(
            lambda x: x.loc[x['holdout_f1_bug'].idxmax()] if x['holdout_f1_bug'].notna().any() else x.iloc[0]
        ).reset_index(drop=True)
        model_best = model_best.sort_values('holdout_f1_bug', ascending=False)

        if not model_best.empty and model_best['holdout_f1_bug'].notna().any():
            plt.figure(figsize=(max(10, len(model_best) * 0.8), 7))
            colors = plt.cm.cividis(np.linspace(0.2, 0.8, len(model_best)))
            bars = plt.bar(model_best['model'], model_best['holdout_f1_bug'].fillna(0), color=colors)

            plt.xlabel('ML Algorithm', fontsize=12)
            plt.ylabel('Holdout F1-Score (Bug)', fontsize=12)
            plt.title(f'F1-Scores by ML Algorithm - {project_name}\n({level.title()} Level, {cv_type_display}, {feature_set_display})', fontsize=14)
            plt.xticks(rotation=45, ha="right", fontsize=10)
            plt.ylim(0, 1.05)
            plt.grid(axis='y', linestyle='--', alpha=0.7)

            for bar in bars:
                yval = bar.get_height()
                if yval > 0:
                    plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.02, f'{yval:.3f}',
                             ha='center', va='bottom', fontsize=9)

            plt.tight_layout()
            plt.savefig(output_dir / "f1_scores_by_ml_algorithm.png", dpi=300, bbox_inches='tight')
            plt.close()
            logging.info(f"ML algorithm plot saved to {output_dir / 'f1_scores_by_ml_algorithm.png'}")
    except Exception as e:
        logging.error(f"Error generating ML algorithm plot: {e}")

    # 2. F1 Scores by Resampling (Holdout)
    try:
        resampling_best = df.groupby('resampling').apply(
            lambda x: x.loc[x['holdout_f1_bug'].idxmax()] if x['holdout_f1_bug'].notna().any() else x.iloc[0]
        ).reset_index(drop=True)
        resampling_best = resampling_best.sort_values('holdout_f1_bug', ascending=False)

        if not resampling_best.empty and resampling_best['holdout_f1_bug'].notna().any():
            plt.figure(figsize=(max(10, len(resampling_best) * 0.8), 7))
            colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(resampling_best)))
            bars = plt.bar(resampling_best['resampling'], resampling_best['holdout_f1_bug'].fillna(0), color=colors)

            plt.xlabel('Resampling Method', fontsize=12)
            plt.ylabel('Holdout F1-Score (Bug)', fontsize=12)
            plt.title(f'F1-Scores by Resampling - {project_name}\n({level.title()} Level, {cv_type_display}, {feature_set_display})', fontsize=14)
            plt.xticks(rotation=45, ha="right", fontsize=10)
            plt.ylim(0, 1.05)
            plt.grid(axis='y', linestyle='--', alpha=0.7)

            for bar in bars:
                yval = bar.get_height()
                if yval > 0:
                    plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.02, f'{yval:.3f}',
                             ha='center', va='bottom', fontsize=9)

            plt.tight_layout()
            plt.savefig(output_dir / "f1_scores_by_resampling.png", dpi=300, bbox_inches='tight')
            plt.close()
            logging.info(f"Resampling plot saved to {output_dir / 'f1_scores_by_resampling.png'}")
    except Exception as e:
        logging.error(f"Error generating resampling plot: {e}")

    # 3. Holdout vs CV Comparison (MCC)
    try:
        # Get best model for each resampling
        comparison_data = df.groupby('resampling').apply(
            lambda x: x.loc[x['holdout_mcc'].idxmax()] if x['holdout_mcc'].notna().any() else x.iloc[0]
        ).reset_index(drop=True)

        if not comparison_data.empty and comparison_data['holdout_mcc'].notna().any():
            fig, ax = plt.subplots(figsize=(max(12, len(comparison_data) * 1.2), 7))

            x = np.arange(len(comparison_data))
            width = 0.35

            cv_mcc = comparison_data['cv_mcc'].fillna(0)
            holdout_mcc = comparison_data['holdout_mcc'].fillna(0)

            bars1 = ax.bar(x - width/2, cv_mcc, width, label='CV MCC', color='#1f77b4')
            bars2 = ax.bar(x + width/2, holdout_mcc, width, label='Holdout MCC', color='#ff7f0e')

            ax.set_xlabel('Resampling Method', fontsize=12)
            ax.set_ylabel('MCC Score', fontsize=12)
            ax.set_title(f'CV vs Holdout MCC Comparison - {project_name}\n({level.title()} Level, {cv_type_display}, {feature_set_display})', fontsize=14)
            ax.set_xticks(x)
            ax.set_xticklabels(comparison_data['resampling'], rotation=45, ha='right')
            ax.legend()
            ax.grid(axis='y', linestyle='--', alpha=0.7)
            ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.5)

            plt.tight_layout()
            plt.savefig(output_dir / "cv_vs_holdout_mcc.png", dpi=300, bbox_inches='tight')
            plt.close()
            logging.info(f"CV vs Holdout plot saved to {output_dir / 'cv_vs_holdout_mcc.png'}")
    except Exception as e:
        logging.error(f"Error generating CV vs Holdout plot: {e}")


def generate_level_summary_report(level, cv_type, feature_set, results_dir):
    """Generate top-level summary report across all projects."""

    all_results = []

    for project_dir in results_dir.iterdir():
        if not project_dir.is_dir() or project_dir.name.startswith('_'):
            continue
        if project_dir.name in ['combined', 'combine']:
            continue

        project_results = collect_project_results(project_dir, cv_type, feature_set)
        for r in project_results:
            r['project'] = project_dir.name
        all_results.extend(project_results)

    if not all_results:
        logging.warning(f"No results found for level {level}, cv_type {cv_type}, feature_set {feature_set}")
        return

    df = pd.DataFrame(all_results)
    cv_type_display = "Temporal CV" if cv_type == "temporal" else "Shuffle CV"
    feature_set_display = "Full Features" if feature_set == "full" else ("Without Go Metrics" if feature_set == "no_go_metrics" else f"Feature Set: {feature_set}")

    # Top 10 by holdout MCC
    top_10 = df.nlargest(10, 'holdout_mcc')

    md_content = f"""# {level.title()} Level Analysis Summary ({cv_type_display}, {feature_set_display})

## Top 10 Best Holdout MCC Results

| Rank | Project | Resampling | Model | CV MCC (Mean ± Std) | Holdout MCC |
|------|---------|------------|-------|---------------------|-------------|
"""

    for idx, (_, row) in enumerate(top_10.iterrows(), 1):
        md_content += f"| {idx} | {row['project']} | {row['resampling']} | {row['model']} | "
        md_content += f"{format_metric_with_std(row['cv_mcc'], row['cv_mcc_std'])} | "
        md_content += f"{format_metric(row['holdout_mcc'])} |\n"

    # Best per project
    md_content += "\n## Best Result per Project (by Holdout MCC)\n\n"
    md_content += "| Project | Resampling | Model | CV F1 (Mean ± Std) | Holdout F1 | Holdout MCC |\n"
    md_content += "|---------|------------|-------|-------------------|------------|-------------|\n"

    best_per_project = df.loc[df.groupby('project')['holdout_mcc'].idxmax()]
    best_per_project = best_per_project.sort_values('holdout_mcc', ascending=False)

    for _, row in best_per_project.iterrows():
        md_content += f"| {row['project']} | {row['resampling']} | {row['model']} | "
        md_content += f"{format_metric_with_std(row['cv_f1_bug'], row['cv_f1_bug_std'])} | "
        md_content += f"{format_metric(row['holdout_f1_bug'])} | "
        md_content += f"{format_metric(row['holdout_mcc'])} |\n"

    # Best per ML algorithm
    md_content += "\n## Best Result per ML Algorithm (by Holdout MCC)\n\n"
    md_content += "| ML Algorithm | Project | Resampling | CV MCC (Mean ± Std) | Holdout MCC |\n"
    md_content += "|--------------|---------|------------|---------------------|-------------|\n"

    best_per_algo = df.loc[df.groupby('model')['holdout_mcc'].idxmax()]
    best_per_algo = best_per_algo.sort_values('holdout_mcc', ascending=False)

    for _, row in best_per_algo.iterrows():
        md_content += f"| {row['model']} | {row['project']} | {row['resampling']} | "
        md_content += f"{format_metric_with_std(row['cv_mcc'], row['cv_mcc_std'])} | "
        md_content += f"{format_metric(row['holdout_mcc'])} |\n"

    md_content += f"\n*Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n"

    # Save to cv_type/feature_set directory
    output_dir = results_dir / "_summary" / cv_type / feature_set
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "results.md"

    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(md_content)
        logging.info(f"Level summary saved to {output_file}")
    except Exception as e:
        logging.error(f"Error saving level summary: {e}")


def run_generate_reports(cli_args):
    """Main entry point for --generate-reports command."""

    levels_to_process = [cli_args.level] if cli_args.level else ALL_LEVELS
    cv_types_to_process = ['shuffle'] if getattr(cli_args, 'shuffle_cv', False) else ['temporal', 'shuffle']

    # Process all feature sets: full and no_go_metrics
    feature_sets_to_process = ['full', 'no_go_metrics']

    for level in levels_to_process:
        logging.info(f"===== Generating reports for LEVEL: {level} =====")

        for cv_type in cv_types_to_process:
            for feature_set in feature_sets_to_process:
                logging.info(f"  Processing CV type: {cv_type}, Feature set: {feature_set}")
                generate_comprehensive_reports(level, cv_type, feature_set)

    logging.info("Report generation complete!")


def generate_important_features_table(level):
    """
    Generates a markdown table of important features for the combined data at a specific level.
    """
    project_name = 'combined' if level in ['commit', 'method'] else 'combine'
    results_dir = get_results_dir(level)

    feature_file = results_dir / project_name / 'feature_selection' / 'combined_results.txt'
    output_file = results_dir / project_name / 'results.md'

    if not feature_file.exists():
        print(f"Feature importance file not found: {feature_file}")
        print(f"Please run feature_select.py for the combined dataset first.")
        print(f"e.g., python feature_select.py --level {level} --project {project_name}")
        return

    features = []
    with open(feature_file, 'r') as f:
        # Skip header lines, which can be variable. Find the separator.
        lines = f.readlines()
        start_index = 0
        for i, line in enumerate(lines):
            if '---' in line:
                start_index = i + 1
                break

        for line in lines[start_index:]:
            line = line.strip()
            if not line:
                continue
            try:
                parts = line.split(':')
                if len(parts) < 2:
                    continue
                feature_name = parts[0].strip()
                score_str = parts[1].strip()
                if score_str.lower() != 'nan':
                    score = float(score_str)
                    features.append((feature_name, score))
            except (ValueError, IndexError) as e:
                print(f"Could not parse line: '{line}'. Error: {e}")

    # Sort by score descending
    features.sort(key=lambda x: x[1], reverse=True)

    # Prepare data for tabulate
    table_data = []
    for i, (feature, score) in enumerate(features):
        table_data.append([i + 1, f"`{feature}`", f"{score:.2f}"])

    headers = ["Rank", "Feature", "Normalized Importance (Avg.)"]
    markdown_table = tabulate.tabulate(table_data, headers=headers, tablefmt="pipe")

    # Append to results.md, creating it if it doesn't exist
    with open(output_file, 'a') as f:
        f.write("\n\n## Aggregate Feature Importance\n\n")
        f.write(markdown_table)
        f.write("\n")

    print(f"Appended feature importance table to {output_file}")

def get_classifier_by_name(method_name):
    """
    Returns a classifier instance and a boolean indicating if it needs scaling.
    """
    needs_scaling = False
    model = None

    if method_name == 'random_forest':
        model = RandomForestClassifier(random_state=42, n_jobs=-1)
    elif method_name == 'xgboost':
        model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    elif method_name == 'lightgbm':
        model = lgb.LGBMClassifier(random_state=42)
    elif method_name == 'logistic_regression':
        model = LogisticRegression(random_state=42, max_iter=1000)
        needs_scaling = True
    elif method_name == 'naive_bayes':
        model = GaussianNB()
    elif method_name == 'gradient_boosting':
        model = GradientBoostingClassifier(random_state=42)
    elif method_name == 'decision_tree':
        model = DecisionTreeClassifier(random_state=42)
    elif method_name == 'catboost':
        model = CatBoostClassifier(random_state=42, verbose=0)
    elif method_name == 'mlp':
        model = MLPClassifier(random_state=42, max_iter=500)
        needs_scaling = True
    elif method_name == 'voting':
        clf1 = LogisticRegression(random_state=42, solver='liblinear', max_iter=1000)
        clf2 = RandomForestClassifier(random_state=42, n_estimators=50)
        clf3 = GaussianNB()
        model = VotingClassifier(estimators=[('lr', clf1), ('rf', clf2), ('gnb', clf3)], voting='soft')
        needs_scaling = True # Because Logistic Regression is included
    elif method_name == 'stacking':
        estimators = [
            ('rf', RandomForestClassifier(n_estimators=10, random_state=42)),
            ('gb', GradientBoostingClassifier(n_estimators=25, random_state=42))
        ]
        final_estimator = LogisticRegression(random_state=42, max_iter=1000)
        model = StackingClassifier(estimators=estimators, final_estimator=final_estimator, cv=5, n_jobs=-1)
        needs_scaling = True # Because the final estimator is Logistic Regression
    else:
        logging.warning(f"Method '{method_name}' not recognized for best feature analysis. Defaulting to RandomForest.")
        model = RandomForestClassifier(random_state=42, n_jobs=-1)
        method_name = 'random_forest'

    return model, needs_scaling, method_name


def run_best_features_analysis(cli_args):
    """
    Finds the optimal number of features for a given project and level
    by iteratively training and evaluating a model with an increasing number of features.
    If --resampling or --methods is 'all', it will iterate through all combinations
    and report the single best result.
    """
    logging.info(f"Starting best feature analysis for project: {cli_args.project}, level: {cli_args.level}")

    # 1. Load Data
    df = load_project_data(cli_args.project, cli_args.level)
    if df is None:
        logging.error("Failed to load data. Exiting.")
        return

    X, y = prepare_features(df, cli_args.level, cli_args=cli_args)
    if X is None or y is None or X.empty:
        logging.error("Failed to prepare features. Exiting.")
        return

    # 2. Get Ranked Features (once, for all iterations)
    logging.info("Ranking features using the 'combine' method...")
    all_methods_for_combine = ['variance', 'chi2', 'rfe', 'lasso', 'rf', 'mi']
    all_results_combine = {}
    for method_name_fs in all_methods_for_combine:
        try:
            _, _, temp_importance_scores = fs.select_features(X.copy(), y.copy(), method=method_name_fs, k=X.shape[1])
            if temp_importance_scores:
                all_results_combine[method_name_fs] = (None, None, temp_importance_scores)
        except Exception as e:
            logging.warning(f"Could not get importance scores for feature selection method '{method_name_fs}': {e}")

    if not all_results_combine:
        logging.error("Could not rank features using any method. Aborting.")
        return

    ranked_features = fs.combine_feature_importance(all_results_combine)
    ranked_feature_names = [feat for feat, score in ranked_features]
    logging.info(f"Feature ranking complete. Total ranked features: {len(ranked_feature_names)}")

    # 3. Determine iteration space for resampling and methods
    resampling_strategies_to_run = []
    if cli_args.resampling == 'all':
        resampling_strategies_to_run = ['none'] + ALL_ACTUAL_RESAMPLING_METHODS
    else:
        resampling_strategies_to_run = [cli_args.resampling]

    methods_to_run = []
    # cli_args.methods is a list.
    if 'all' in cli_args.methods:
        methods_to_run = ALL_CLASSIFIER_FUNCTION_NAMES
    else:
        methods_to_run = cli_args.methods

    overall_run_results = []
    total_combinations = len(resampling_strategies_to_run) * len(methods_to_run)
    logging.info(f"Total combinations to test: {total_combinations} ({len(resampling_strategies_to_run)} resampling x {len(methods_to_run)} methods)")

    # 4. Main loop for evaluation
    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeRemainingColumn(),
        TimeElapsedColumn(),
        transient=False # Keep it on screen after finishing
    ) as progress:
        main_task = progress.add_task("[bold green]Analyzing combinations...", total=total_combinations)

        for resampling_strategy in resampling_strategies_to_run:
            for method_name in methods_to_run:
                progress.update(main_task, description=f"[green]Testing: {resampling_strategy} / {method_name}")

                # Get model for this iteration
                model, needs_scaling, final_method_name = get_classifier_by_name(method_name)

                # Create a temporary args object for evaluate_feature_subset
                temp_cli_args = argparse.Namespace(**vars(cli_args))
                temp_cli_args.resampling = resampling_strategy

                # This list will hold results for this specific combination
                # (baseline + each k subset)
                current_combo_evaluations = []

                # Evaluate with all original features as a baseline
                all_features_metrics = evaluate_feature_subset(X, y, model, temp_cli_args, needs_scaling=needs_scaling)
                if all_features_metrics:
                    current_combo_evaluations.append({'k': 'all_pre_ranking', 'metrics': all_features_metrics, 'features': list(X.columns)})

                # Evaluate ranked subsets
                for k in range(1, len(ranked_feature_names) + 1):
                    current_features = ranked_feature_names[:k]
                    X_subset = X[current_features]

                    avg_metrics = evaluate_feature_subset(X_subset, y, model, temp_cli_args, needs_scaling=needs_scaling)
                    if avg_metrics:
                        current_combo_evaluations.append({'k': k, 'metrics': avg_metrics, 'features': current_features})

                # Find the best result for *this combination*
                if current_combo_evaluations:
                    # Handle cases where metrics might be None or empty
                    valid_evaluations = [e for e in current_combo_evaluations if e.get('metrics') and e['metrics'].get('f1') is not None]
                    if valid_evaluations:
                        best_result_for_combo = max(valid_evaluations, key=lambda x: x['metrics']['f1'])

                        # Store the summary of the best result for this combination
                        overall_run_results.append({
                            'resampling': resampling_strategy,
                            'method': final_method_name,
                            'best_k': best_result_for_combo['k'],
                            'metrics': best_result_for_combo['metrics'],
                            'features': best_result_for_combo['features']
                        })

                progress.update(main_task, advance=1)

    # 5. Find and report the single best result from all combinations
    if not overall_run_results:
        logging.error("No results were generated from the entire analysis.")
        return

    final_best_result = max(overall_run_results, key=lambda x: x['metrics'].get('f1', 0))

    # Print the final report
    print("\n\n--- Overall Best Configuration Found ---")
    print(f"Project: {cli_args.project}, Level: {cli_args.level}")
    print(f"Resampling Method: {final_best_result['resampling']}")
    print(f"ML Model: {final_best_result['method']}")
    print(f"Best number of features (k): {final_best_result['best_k']}")
    print(f"Best F1-score (for bugs): {final_best_result['metrics']['f1']:.4f}")
    print(f"Accuracy: {final_best_result['metrics']['accuracy']:.4f}")
    print(f"Precision (for bugs): {final_best_result['metrics']['precision']:.4f}")
    print(f"Recall (for bugs): {final_best_result['metrics']['recall']:.4f}")
    print("Best feature set:")
    for feature in final_best_result['features']:
        print(f"- {feature}")


    # 6. Save all summary results to a single file
    resampling_name_for_file = cli_args.resampling if cli_args.resampling != 'all' else 'all'
    method_name_for_file = 'all' if 'all' in cli_args.methods else '_'.join(cli_args.methods)

    results_dir = get_results_dir(cli_args.level) / cli_args.project / f"analysis_best_features_{resampling_name_for_file}_{method_name_for_file}"
    results_dir.mkdir(parents=True, exist_ok=True)
    results_file = results_dir / "best_features_analysis_summary.json"

    # Add the overall best to the saved data for easy identification
    output_data = {
        'overall_best_configuration': final_best_result,
        'all_combination_results': overall_run_results
    }

    with open(results_file, 'w') as f:
        json.dump(convert_numpy_to_list_recursive(output_data), f, indent=4)

    logging.info(f"Best features analysis summary saved to {results_file}")


def evaluate_feature_subset(X, y, model, cli_args, needs_scaling=False):
    """
    Evaluates a subset of features using k-fold cross-validation.
    """
    kfold = KFold(n_splits=cli_args.folds, shuffle=True, random_state=42)
    fold_metrics = []

    for train_idx, test_idx in kfold.split(X, y):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        # Apply scaling if needed
        if needs_scaling:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

        # Apply resampling
        resampler = get_resampling_method(cli_args.resampling)
        if resampler:
            try:
                X_train_resampled, y_train_resampled = resampler.fit_resample(X_train, y_train)
            except Exception as e:
                logging.warning(f"Resampling with {cli_args.resampling} failed for a fold: {e}")
                X_train_resampled, y_train_resampled = X_train, y_train # Fallback to original data for this fold
        else:
            X_train_resampled, y_train_resampled = X_train, y_train

        # Check for single class in training data *after* resampling
        if pd.Series(y_train_resampled).nunique() < 2:
            logging.warning(
                f"Skipping a fold for {type(model).__name__} with resampling '{cli_args.resampling}' "
                f"because training data for this fold has only one class after resampling. "
                f"Class distribution: {pd.Series(y_train_resampled).value_counts().to_dict()}"
            )
            continue # Skip this fold

        # Clone the model to ensure a fresh state for each fold
        current_model = clone(model)

        try:
            current_model.fit(X_train_resampled, y_train_resampled)
            y_pred = current_model.predict(X_test)
            report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

            metrics = {
                'accuracy': report.get('accuracy', 0),
                'precision': report.get('1', {}).get('precision', 0), # for class 1 (buggy)
                'recall': report.get('1', {}).get('recall', 0),
                'f1': report.get('1', {}).get('f1-score', 0)
            }
            fold_metrics.append(metrics)
        except Exception as e:
            logging.error(f"Model fitting or prediction failed for a fold. Model: {type(current_model).__name__}, Resampling: {cli_args.resampling}. Error: {e}")

    if not fold_metrics:
        logging.warning(f"All folds were skipped or failed for {type(model).__name__} with resampling '{cli_args.resampling}'. No average metrics can be computed.")
        return None

    return pd.DataFrame(fold_metrics).mean().to_dict()

def find_best_feature_set(cli_args):
    """
    Finds the optimal number of features for a given project and level
    by iteratively training and evaluating a model with an increasing number of features.
    This uses a default, non-optimized model for speed.
    """
    logging.info(f"Finding best feature set for project: {cli_args.project}, level: {cli_args.level}")

    df = load_project_data(cli_args.project, cli_args.level)
    if df is None:
        logging.error("Failed to load data for best feature search. Exiting.")
        return None

    X, y = prepare_features(df, cli_args.level, cli_args=cli_args)
    if X is None or y is None or X.empty:
        logging.error("Failed to prepare features for best feature search. Exiting.")
        return None

    logging.info("Ranking features using the 'combine' method for feature search...")
    all_methods_for_combine = ['variance', 'chi2', 'rfe', 'lasso', 'rf', 'mi']
    all_results_combine = {}
    for method_name_fs in all_methods_for_combine:
        try:
            _, _, temp_importance_scores = fs.select_features(X.copy(), y.copy(), method=method_name_fs, k=X.shape[1])
            if temp_importance_scores:
                all_results_combine[method_name_fs] = (None, None, temp_importance_scores)
        except Exception as e:
            logging.warning(f"Could not get importance scores for feature selection method '{method_name_fs}': {e}")

    if not all_results_combine:
        logging.error("Could not rank features using any method. Aborting feature search.")
        return None

    ranked_features = fs.combine_feature_importance(all_results_combine)
    ranked_feature_names = [feat for feat, score in ranked_features]
    logging.info(f"Feature ranking complete. Total ranked features: {len(ranked_feature_names)}")

    # Use a default, fast model for this search (RandomForest is a good choice)
    model, needs_scaling, _ = get_classifier_by_name('random_forest')

    evaluation_results = []
    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        transient=False
    ) as progress:
        task = progress.add_task("[cyan]Evaluating feature subsets...", total=len(ranked_feature_names))
        for k in range(1, len(ranked_feature_names) + 1):
            current_features = ranked_feature_names[:k]
            X_subset = X[current_features]

            avg_metrics = evaluate_feature_subset(X_subset, y, model, cli_args, needs_scaling=needs_scaling)
            if avg_metrics:
                evaluation_results.append({'k': k, 'metrics': avg_metrics, 'features': current_features})
            progress.update(task, advance=1)

    if not evaluation_results:
        logging.error("No feature subsets could be evaluated.")
        return None

    # Find the best result based on the F1 score from the evaluations
    best_result = max(evaluation_results, key=lambda x: x['metrics'].get('f1', 0))

    return best_result['features']


def run_cpdp_workflow(cli_args):
    """Orchestrates the Cross-Project Defect Prediction (CPDP) workflow."""
    logging.info("--- Starting Cross-Project Defect Prediction (CPDP) Workflow ---")
    logging.info(f"Source: {cli_args.source}, Destination: {cli_args.destination}, Level: {cli_args.level}")

    if cli_args.resampling == 'all' or (isinstance(cli_args.methods, list) and 'all' in cli_args.methods):
        logging.error("CPDP mode currently does not support 'all' for --resampling or --methods. Please specify one of each.")
        return

    methods_to_run = cli_args.methods
    if not isinstance(methods_to_run, list) or len(methods_to_run) == 0:
        logging.error("No method specified for CPDP analysis. Please use --methods.")
        return

    # Load Data
    logging.info(f"Loading source data for '{cli_args.source}'...")
    source_df = load_project_data(cli_args.source, cli_args.level)
    if source_df is None:
        logging.error(f"Failed to load source data for '{cli_args.source}'. Aborting CPDP.")
        return

    logging.info(f"Loading destination data for '{cli_args.destination}'...")
    destination_df = load_project_data(cli_args.destination, cli_args.level)
    if destination_df is None:
        logging.error(f"Failed to load destination data for '{cli_args.destination}'. Aborting CPDP.")
        return

    print_class_distribution(f"Source: {cli_args.source}", source_df['is_bug'])
    print_class_distribution(f"Destination: {cli_args.destination}", destination_df['is_bug'])

    # Prepare Features & Align Columns
    X_train, y_train = prepare_features(source_df, cli_args.level, cli_args=cli_args)
    X_test, y_test = prepare_features(destination_df, cli_args.level, cli_args=cli_args)

    if X_train is None or X_test is None or X_train.empty:
        logging.error("Feature preparation failed. Aborting.")
        return

    if len(y_train.unique()) < 2:
        logging.error(f"Source project '{cli_args.source}' has only one class. Cannot train a model. Aborting.")
        return

    # Align columns - crucial for CPDP
    train_cols = X_train.columns
    test_cols = X_test.columns
    missing_in_test = set(train_cols) - set(test_cols)
    if missing_in_test:
        logging.warning(f"Destination data is missing {len(missing_in_test)} columns present in source. Filling with 0.")
        for c in missing_in_test:
            X_test[c] = 0
    extra_in_test = set(test_cols) - set(train_cols)
    if extra_in_test:
        logging.warning(f"Destination data has {len(extra_in_test)} extra columns not in source. They will be removed.")
        X_test = X_test.drop(columns=list(extra_in_test))
    X_test = X_test[train_cols] # Ensure order is the same

    # Run for all specified methods
    for method_to_run in methods_to_run:
        logging.info(f"--- Running CPDP for method: {method_to_run} ---")
        resampling_strategy = cli_args.resampling if cli_args.resampling != 'none' else None

        # Create a unique output directory for this CPDP run
        cpdp_results_base_dir = get_results_dir(cli_args.level, cpdp_mode=True, source_project=cli_args.source, destination_project=cli_args.destination)
        resampling_name_for_path = resampling_strategy if resampling_strategy else 'none'
        instance_selection_name = f"_{cli_args.instance_selection}_k{cli_args.k_neighbors}" if cli_args.instance_selection else ""
        analysis_dir_name = f"analysis_{resampling_name_for_path}{instance_selection_name}_{method_to_run}" + ("_optimized" if cli_args.optimize else "")
        output_dir = cpdp_results_base_dir / analysis_dir_name
        output_dir.mkdir(parents=True, exist_ok=True)
        logging.info(f"Results will be saved in: {output_dir}")

        # The core evaluation
        results = perform_cpdp_evaluation(
            X_train.copy(), y_train.copy(), X_test.copy(), y_test.copy(),
            method_to_run=method_to_run,
            resampling_strategy=resampling_strategy,
            optimize=cli_args.optimize,
            project_name=f"{cli_args.source}-to-{cli_args.destination}",
            level=cli_args.level,
            output_dir=output_dir,
            instance_selection_strategy=cli_args.instance_selection,
            k_neighbors=cli_args.k_neighbors
        )

        if results:
            logging.info(f"--- CPDP for method '{method_to_run}' Complete ---")
            results['source_project'] = cli_args.source
            results['destination_project'] = cli_args.destination
            results['level'] = cli_args.level
            results['resampling'] = resampling_name_for_path
            results['method'] = method_to_run
            results['optimized'] = cli_args.optimize

            # Save and plot
            results_path = output_dir / "cpdp_results.json"
            try:
                with open(results_path, 'w', encoding='utf-8') as f:
                    json.dump(convert_numpy_to_list_recursive(results), f, indent=4)
                logging.info(f"CPDP results saved to {results_path}")
            except Exception as e:
                logging.error(f"Failed to save CPDP results: {e}")

            if 'y_test_fold' in results and 'y_prob_fold' in results:
                roc_data = {
                    method_to_run: [{
                        'y_test': results['y_test_fold'],
                        'y_prob': results['y_prob_fold']
                    }]
                }
                plot_roc_curves(roc_data, f"CPDP: {cli_args.source} to {cli_args.destination}", cli_args.level, output_dir)
        else:
            logging.error(f"--- CPDP for method '{method_to_run}' Failed ---")


def perform_cpdp_evaluation(X_train, y_train, X_test, y_test, method_to_run, resampling_strategy, optimize, project_name, level, output_dir, instance_selection_strategy=None, k_neighbors=None):
    """
    Performs a single cross-project evaluation with mandatory scaling.
    This is the core workhorse for both CPDP and LOPO.
    """
    logging.info(f"Starting model evaluation for '{project_name}' using '{method_to_run}'")

    # 1. Mandatory Z-Score Normalization
    logging.info("Applying Z-score normalization (fitting on train, transforming train and test).")
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)

    # 2. Instance Selection (Optional) - Applied before resampling
    X_train_processed = X_train_scaled
    y_train_processed = y_train
    if instance_selection_strategy == 'nn_filter':
        if k_neighbors is None or k_neighbors <= 0:
            logging.error("k_neighbors must be a positive integer for nn_filter. Skipping instance selection.")
        else:
            logging.info(f"Applying Nearest Neighbor instance selection with k={k_neighbors}...")
            X_train_filtered, y_train_filtered = apply_nearest_neighbor_filter(
                X_train_scaled, y_train, X_test_scaled, k_neighbors
            )
            X_train_processed = X_train_filtered
            y_train_processed = y_train_filtered

    # 3. Resampling on Training Data (potentially filtered)
    if resampling_strategy:
        logging.info(f"Applying resampling strategy '{resampling_strategy}'...")
        resampler = get_resampling_method(resampling_strategy)
        if resampler:
            try:
                # Use the result from instance selection (or the scaled data if no selection)
                X_train_resampled, y_train_resampled = resampler.fit_resample(X_train_processed, y_train_processed)
                print_class_distribution(f"Training data after resampling ({resampling_strategy})", y_train_resampled)
                X_train_processed, y_train_processed = X_train_resampled, y_train_resampled
            except Exception as e:
                logging.error(f"Error during resampling with {resampling_strategy}: {e}")
                return None

    # 4. Get the correct analysis function for the model
    all_classifier_funcs = {
        'naive_bayes': analyze_with_naive_bayes, 'xgboost': analyze_with_xgboost,
        'random_forest': analyze_with_random_forest,
        'logistic_regression': analyze_with_logistic_regression, 'catboost': analyze_with_catboost,
        'lightgbm': analyze_with_lightgbm, 'gradient_boosting': analyze_with_gradient_boosting,
        'decision_tree': analyze_with_decision_tree, 'voting': analyze_with_voting,
        'mlp': analyze_with_mlp, 'stacking': analyze_with_stacking
    }
    clf_func = all_classifier_funcs.get(method_to_run)
    if not clf_func:
        logging.error(f"Unknown method '{method_to_run}'. Aborting evaluation.")
        return None

    # 5. Run analysis.
    # We pass needs_scaling=False because we have already manually scaled the data.
    # This prevents _run_analysis from scaling it again.
    metrics_with_model = clf_func(
        X_train_processed, X_test_scaled, y_train_processed, y_test,
        project_name=project_name,
        output_dir=output_dir,
        optimize=optimize,
        level=level,
        resampling_method=resampling_strategy
    )

    if not metrics_with_model or not metrics_with_model.metrics:
        logging.error("Model analysis failed to return metrics.")
        return None

    return metrics_with_model.metrics


def apply_nearest_neighbor_filter(X_train, y_train, X_test, k):
    """
    Filters the training data based on proximity to test data using Nearest Neighbors.

    Args:
        X_train (pd.DataFrame): Normalized training feature data.
        y_train (pd.Series): Training target data.
        X_test (pd.DataFrame): Normalized testing feature data.
        k (int): The number of nearest neighbors to find for each test instance.

    Returns:
        (pd.DataFrame, pd.Series): The filtered X_train and y_train.
    """
    logging.info(f"Original training set size: {len(X_train)} instances.")

    # Use NearestNeighbors to find the k closest training instances for each test instance
    nn = NearestNeighbors(n_neighbors=k, algorithm='auto', metric='euclidean', n_jobs=-1)
    nn.fit(X_train)

    # We get the indices of the k neighbors in X_train for each point in X_test
    _, indices = nn.kneighbors(X_test)

    # Flatten the array of indices and get the unique set
    unique_indices = np.unique(indices.flatten())

    X_train_filtered = X_train.iloc[unique_indices]
    y_train_filtered = y_train.iloc[unique_indices]

    logging.info(f"Filtered training set size: {len(X_train_filtered)} instances.")

    return X_train_filtered, y_train_filtered


def run_lopo_workflow(cli_args):
    """Orchestrates the Leave-One-Project-Out (LOPO) cross-validation workflow."""
    logging.info("--- Starting Leave-One-Project-Out (LOPO) Workflow ---")
    level = cli_args.level
    level_data_dir = get_data_dir(level)

    # Get all available projects
    all_projects = sorted([d.name for d in level_data_dir.iterdir() if d.is_dir() and not d.name.startswith('_')])
    if len(all_projects) < 2:
        logging.error(f"LOPO requires at least 2 projects, but only {len(all_projects)} found for level '{level}'. Aborting.")
        return

    logging.info(f"Found {len(all_projects)} projects for LOPO at level '{level}': {all_projects}")

    methods_to_run = ALL_CLASSIFIER_FUNCTION_NAMES if 'all' in cli_args.methods else cli_args.methods
    resampling_strategy = cli_args.resampling if cli_args.resampling != 'none' else None
    resampling_name_for_path = cli_args.resampling

    # Create base output directory for this LOPO run
    lopo_base_dir = get_results_dir(level, lopo_mode=True)
    instance_selection_name = f"_{cli_args.instance_selection}_k{cli_args.k_neighbors}" if cli_args.instance_selection else ""
    run_dir_name = f"lopo_{resampling_name_for_path}{instance_selection_name}" + ("_optimized" if cli_args.optimize else "")
    output_dir = lopo_base_dir / run_dir_name
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.info(f"LOPO results will be saved in: {output_dir}")

    final_summary = {}

    for method in methods_to_run:
        logging.info(f"===== Running LOPO for Method: {method} =====")
        fold_results = []

        with Progress(*Progress.get_default_columns(), transient=True) as progress:
            task = progress.add_task(f"[cyan]LOPO Folds for {method}", total=len(all_projects))

            for i, held_out_project in enumerate(all_projects):
                progress.update(task, description=f"[cyan]LOPO for {method} | Testing on: {held_out_project}")

                source_projects = [p for p in all_projects if p != held_out_project]

                # Load and combine source data
                source_dfs = [load_project_data(p, level) for p in source_projects]
                source_dfs = [df for df in source_dfs if df is not None and not df.empty]
                if not source_dfs:
                    logging.error(f"Could not load any source data for fold {i+1} (testing on {held_out_project}). Skipping fold.")
                    progress.update(task, advance=1)
                    continue
                train_df = pd.concat(source_dfs, ignore_index=True)

                # Load destination data
                test_df = load_project_data(held_out_project, level)
                if test_df is None or test_df.empty:
                    logging.error(f"Could not load test data for {held_out_project}. Skipping fold.")
                    progress.update(task, advance=1)
                    continue

                # Prepare features
                X_train, y_train = prepare_features(train_df, level, cli_args=cli_args)
                X_test, y_test = prepare_features(test_df, level, cli_args=cli_args)

                if X_train is None or X_test is None or X_train.empty:
                    logging.error("Feature preparation failed for a fold. Skipping.")
                    progress.update(task, advance=1)
                    continue
                if len(y_train.unique()) < 2:
                    logging.error(f"Combined training data for fold {i+1} has only one class. Skipping.")
                    progress.update(task, advance=1)
                    continue

                # Align columns
                train_cols = X_train.columns
                test_cols = X_test.columns
                missing_in_test = set(train_cols) - set(test_cols)
                for c in missing_in_test: X_test[c] = 0
                extra_in_test = set(test_cols) - set(train_cols)
                X_test = X_test.drop(columns=list(extra_in_test), errors='ignore')
                X_test = X_test[train_cols]

                # Run evaluation
                fold_metric = perform_cpdp_evaluation(
                    X_train.copy(), y_train.copy(), X_test.copy(), y_test.copy(),
                    method_to_run=method,
                    resampling_strategy=resampling_strategy,
                    optimize=cli_args.optimize,
                    project_name=f"LOPO-fold-{i+1}",
                    level=level,
                    output_dir=output_dir / "temp", # Temp dir, not used for saving fold-specific plots
                    instance_selection_strategy=cli_args.instance_selection,
                    k_neighbors=cli_args.k_neighbors
                )

                if fold_metric:
                    fold_metric['held_out_project'] = held_out_project
                    fold_results.append(fold_metric)

                progress.update(task, advance=1)

        # After all folds for a method
        if not fold_results:
            logging.error(f"No results generated for method '{method}' in LOPO run. Skipping summary.")
            continue

        # Save detailed fold results for the method
        method_fold_path = output_dir / f"{method}_lopo_fold_results.json"
        with open(method_fold_path, 'w') as f:
            json.dump(convert_numpy_to_list_recursive(fold_results), f, indent=4)
        logging.info(f"Saved detailed LOPO fold results for '{method}' to {method_fold_path}")

        # Calculate and store summary stats
        metrics_df = pd.DataFrame(fold_results)
        mean_metrics = metrics_df.mean(numeric_only=True)
        std_metrics = metrics_df.std(numeric_only=True)

        final_summary[method] = {
            'mean': mean_metrics.to_dict(),
            'std': std_metrics.to_dict()
        }

    # After all methods
    logging.info("===== LOPO Final Summary =====")
    summary_path = output_dir / "lopo_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(convert_numpy_to_list_recursive(final_summary), f, indent=4)
    logging.info(f"Final LOPO summary saved to {summary_path}")

    # Print a nice summary table to console
    table_data = []
    headers = ["ML Method", "Mean F1-Score (Bug)", "Std Dev F1-Score", "Mean Accuracy", "Std Dev Accuracy", "Mean AUC", "Std Dev AUC"]
    for method, stats in final_summary.items():
        table_data.append([
            method,
            f"{stats['mean'].get('f1_1', 0):.4f}",
            f"{stats['std'].get('f1_1', 0):.4f}",
            f"{stats['mean'].get('accuracy', 0):.4f}",
            f"{stats['std'].get('accuracy', 0):.4f}",
            f"{stats['mean'].get('auc', 0):.4f}",
            f"{stats['std'].get('auc', 0):.4f}"
        ])

    # Sort by mean F1-score
    table_data.sort(key=lambda row: float(row[1]), reverse=True)

    print("\n" + tabulate.tabulate(table_data, headers=headers, tablefmt="grid"))


def print_final_metrics(results_dict):
    """Prints a formatted summary of the final metrics to the console."""
    if not results_dict:
        print("No results to display.")
        return

    project = results_dict.get('project', 'N/A')
    level = results_dict.get('level', 'N/A')
    resampling = results_dict.get('resampling', 'N/A')

    print("\n" + "="*60)
    print(f"Analysis Summary for Project: {project} ({level} level)")
    print(f"Resampling Strategy: {resampling}")
    print("="*60)

    for model_name, metrics in results_dict.items():
        if model_name in ['project', 'level', 'resampling', 'error', 'cv_type']:
            continue

        if isinstance(metrics, dict):
            # Handle temporal_validation separately
            if model_name == 'temporal_validation':
                cv_type = results_dict.get('cv_type', 'temporal')
                cv_label = "Shuffle CV" if cv_type == 'shuffle' else "Temporal CV"
                print(f"\n--- Validation Summary ({cv_label}) ---")
                table_data = []
                headers = ["Check", "Result"]

                table_data.append(["CV Type", cv_label])

                holdout_valid = metrics.get('holdout_temporal_valid', False)
                n_valid_folds = metrics.get('n_valid_folds', 0)

                table_data.append(["Holdout Temporal Valid", "✓ Yes" if holdout_valid else "✗ No"])
                if cv_type == 'temporal':
                    table_data.append(["Valid Folds (temporal + group)", str(n_valid_folds)])
                else:
                    table_data.append(["Folds Used", str(n_valid_folds)])

                print(tabulate.tabulate(table_data, headers=headers, tablefmt="grid"))
                continue

            print(f"\n--- Model: {model_name.replace('_', ' ').title()} ---")

            # Define the order of metrics to print
            metrics_to_print = [
                ('accuracy', 'Accuracy'),
                ('f1_1', 'F1 Score (Bug)'),
                ('precision_1', 'Precision (Bug)'),
                ('recall_1', 'Recall (Bug)'),
                ('auc', 'ROC-AUC'),
                ('pr_auc', 'PR-AUC'),  # Precision-Recall AUC
                ('mcc', 'MCC'),  # Matthews Correlation Coefficient
                ('f1_0', 'F1 Score (Clean)'),
                ('precision_0', 'Precision (Clean)'),
                ('recall_0', 'Recall (Clean)'),
            ]

            # Check if holdout results exist
            has_holdout = any(f'holdout_{key}' in metrics for key, _ in metrics_to_print)

            if has_holdout:
                # Show both CV and Holdout results
                table_data = []
                headers = ["Metric", "CV (Mean ± Std)", "Hold-out"]

                for key, name in metrics_to_print:
                    cv_value = metrics.get(key)
                    cv_std = metrics.get(f'{key}_std')
                    holdout_value = metrics.get(f'holdout_{key}')

                    if cv_value is not None:
                        if cv_std is not None:
                            cv_str = f"{cv_value:.4f} ± {cv_std:.4f}"
                        else:
                            cv_str = f"{cv_value:.4f}"
                    else:
                        cv_str = "-"

                    holdout_str = f"{holdout_value:.4f}" if holdout_value is not None else "-"
                    table_data.append([name, cv_str, holdout_str])
            else:
                # Only CV results available
                table_data = []
                headers = ["Metric", "CV (Mean ± Std)"]

                for key, name in metrics_to_print:
                    value = metrics.get(key)
                    std_value = metrics.get(f'{key}_std')
                    if value is not None:
                        if std_value is not None:
                            table_data.append([name, f"{value:.4f} ± {std_value:.4f}"])
                        else:
                            table_data.append([name, f"{value:.4f}"])

            if 'error' in metrics:
                 table_data.append(["Error", metrics.get('error', '')])

            print(tabulate.tabulate(table_data, headers=headers, tablefmt="grid"))

    print("\n" + "="*60)


if __name__ == "__main__":
    main()
