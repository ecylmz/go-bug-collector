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
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.naive_bayes import GaussianNB
# from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc, roc_auc_score
from sklearn.tree import DecisionTreeClassifier # Added
from sklearn.neural_network import MLPClassifier # Added
from sklearn.neighbors import NearestNeighbors
from catboost import CatBoostClassifier
from imblearn.over_sampling import SMOTE, RandomOverSampler, ADASYN, BorderlineSMOTE
from imblearn.under_sampling import RandomUnderSampler, NearMiss, TomekLinks
from imblearn.combine import SMOTETomek, SMOTEENN
from pathlib import Path
import contextlib
import tabulate
from tqdm.rich import tqdm
from sklearn.utils import shuffle
from scipy.stats import norm
from scipy.stats import friedmanchisquare
import scikit_posthocs as sp # Added for Nemenyi test
import autorank
import feature_select as fs # Added for dynamic feature selection
import warnings
warnings.filterwarnings('ignore')
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn, TimeElapsedColumn, MofNCompleteColumn
from sklearn.base import clone

# --- Global Definitions ---
ALL_LEVELS = ['commit', 'file', 'method']

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

def load_project_data(project_name, level):
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

    bugs_df['is_bug'] = 1
    non_bugs_df['is_bug'] = 0

    combined_df = pd.concat([bugs_df, non_bugs_df], ignore_index=True)
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


def get_metrics(y_test, y_pred_proba, feature_importance=None):
    if len(y_pred_proba.shape) > 1:
        y_pred = (y_pred_proba[:, 1] >= 0.5).astype(int)
        y_prob = y_pred_proba[:, 1]
    else:
        y_pred = (y_pred_proba >= 0.5).astype(int) # Assuming binary output if not proba
        y_prob = y_pred_proba

    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    auc_score_val = roc_auc_score(y_test, y_prob) if y_prob is not None and len(np.unique(y_test)) > 1 else 0.0

    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision_0': report.get('0', {}).get('precision', 0),
        'recall_0': report.get('0', {}).get('recall', 0),
        'f1_0': report.get('0', {}).get('f1-score', 0),
        'precision_1': report.get('1', {}).get('precision', 0),
        'recall_1': report.get('1', {}).get('recall', 0),
        'f1_1': report.get('1', {}).get('f1-score', 0),
        'auc': auc_score_val,
        # ADD y_test and y_prob here for ROC data collection
        'y_test_fold': y_test.tolist() if hasattr(y_test, 'tolist') else list(y_test),
        'y_prob_fold': y_prob.tolist() if hasattr(y_prob, 'tolist') else list(y_prob)
    }
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
        return float(data)
    if isinstance(data, np.bool_):
        return bool(data)
    if isinstance(data, np.str_): # Handle numpy strings
        return str(data)
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

def _optimize_model(model, param_grid, X_train, y_train, model_name):
    logging.info(f"Optimizing {model_name}...")
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='f1', n_jobs=-1, verbose=0) # cv=5 for speed
    grid_search.fit(X_train, y_train)
    logging.info(f"Best parameters for {model_name}: {grid_search.best_params_}")
    logging.info(f"Best CV score for {model_name}: {grid_search.best_score_:.4f}")
    return grid_search.best_estimator_, grid_search.best_params_, grid_search.best_score_

def optimize_naive_bayes(X_train, y_train):
    model, params, score = _optimize_model(GaussianNB(), {'var_smoothing': np.logspace(0,-9, num=10)}, X_train, y_train, "Naive Bayes")
    return model, params, score

def optimize_xgboost(X_train, y_train):
    param_grid = {
        'n_estimators': [100, 200],
        'learning_rate': [0.01, 0.1],
        'max_depth': [3, 5, 7],
        'subsample': [0.7, 1.0],
        'colsample_bytree': [0.7, 1.0],
        'gamma': [0, 0.1, 0.2],
        'reg_alpha': [0, 0.1],
        'reg_lambda': [1, 1.5]
    }
    model, params, score = _optimize_model(xgb.XGBClassifier(objective='binary:logistic', use_label_encoder=False, eval_metric='logloss', random_state=42),
                           param_grid,
                           X_train, y_train, "XGBoost")
    return model, params, score

def optimize_random_forest(X_train, y_train):
    param_grid = {
        'n_estimators': [100, 200],
        'max_features': ['sqrt', 'log2'],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
        'criterion': ['gini', 'entropy']
    }
    model, params, score = _optimize_model(RandomForestClassifier(class_weight='balanced', random_state=42),
                           param_grid, X_train, y_train, "Random Forest")
    return model, params, score

def optimize_lightgbm(X_train, y_train):
    param_grid = {
        'n_estimators': [100, 200],
        'learning_rate': [0.01, 0.1],
        'num_leaves': [31, 50],
        'max_depth': [-1, 10, 20],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0],
        'reg_alpha': [0, 0.1],
        'reg_lambda': [0, 0.1]
    }
    model, params, score = _optimize_model(lgb.LGBMClassifier(class_weight='balanced', verbose=-1, random_state=42),
                           param_grid,
                           X_train, y_train, "LightGBM")
    return model, params, score

def optimize_catboost(X_train, y_train):
    param_grid = {
        'iterations': [100, 200],
        'learning_rate': [0.01, 0.1],
        'depth': [4, 6, 8],
        'l2_leaf_reg': [1, 3, 5],
        'rsm': [0.7, 1.0], # Equivalent to colsample_bytree
        'subsample': [0.8, 1.0]
    }
    model, params, score = _optimize_model(CatBoostClassifier(verbose=0, auto_class_weights='Balanced', random_state=42),
                           param_grid,
                           X_train, y_train, "CatBoost")
    return model, params, score

def optimize_logistic_regression(X_train, y_train):
    # Using a list of dicts to handle solver-penalty compatibility correctly.
    param_grid = [
        {
            'solver': ['liblinear', 'saga'],
            'penalty': ['l1'],
            'C': [0.1, 1.0, 10.0]
        },
        {
            'solver': ['lbfgs', 'liblinear', 'saga'],
            'penalty': ['l2'],
            'C': [0.1, 1.0, 10.0]
        }
    ]
    model, params, score = _optimize_model(LogisticRegression(class_weight='balanced', random_state=42, max_iter=2000),
                           param_grid, X_train, y_train, "Logistic Regression")
    return model, params, score

def optimize_gradient_boosting(X_train, y_train):
    # Expanded parameter grid
    param_grid = {
        'n_estimators': [100, 200],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7],
        'subsample': [0.8, 1.0],
        'max_features': ['sqrt', 'log2']
    }
    model, params, score = _optimize_model(GradientBoostingClassifier(random_state=42),
                           param_grid, X_train, y_train, "Gradient Boosting")
    return model, params, score

def optimize_decision_tree(X_train, y_train):
    # Expanded parameter grid
    param_grid = {
        'criterion': ['gini', 'entropy'],
        'max_depth': [5, 10, 20, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2', None]
    }
    model, params, score = _optimize_model(DecisionTreeClassifier(class_weight='balanced', random_state=42),
                           param_grid, X_train, y_train, "Decision Tree")
    return model, params, score

def optimize_mlp(X_train, y_train):
    # Expanded base parameters for MLP optimization
    param_grid = {
        'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50)],
        'activation': ['relu', 'tanh'],
        'alpha': [0.0001, 0.001, 0.01],
        'learning_rate_init': [0.001, 0.01]
    }

    # MLP instance for GridSearchCV. early_stopping defaults to False.
    # It will be enabled by GridSearchCV if 'early_stopping': [True] is in param_grid.
    mlp_for_grid = MLPClassifier(random_state=42, max_iter=2000)

    if len(y_train) < 20: # Threshold: validation_fraction (0.1) * len(y_train) must be >= 2 (n_classes)
        logging.warning(
            f"Optimizing MLP: Training data size ({len(y_train)}) is too small for reliable early stopping validation "
            f"with validation_fraction=0.1. Early stopping parameters will not be included in the grid search."
        )
        # param_grid remains as base_param_grid; early_stopping will effectively be False for all trials.
    else:
        # Sufficient data, add early stopping related parameters to the grid search
        logging.info(f"Optimizing MLP: Training data size ({len(y_train)}) is sufficient. Including early stopping parameters in grid search.")
        param_grid['early_stopping'] = [True] # Only try True, as False is default
        param_grid['validation_fraction'] = [0.1] # Could be tuned, fixed for now
        param_grid['n_iter_no_change'] = [10]    # Could be tuned, fixed for now

    model, params, score = _optimize_model(mlp_for_grid, param_grid, X_train, y_train, "MLP")
    return model, params, score

def optimize_voting(X_train, y_train):
    # Expanded grid for Voting Classifier
    estimators = [
        ('rf', RandomForestClassifier(random_state=42, class_weight='balanced')),
        ('xgb', xgb.XGBClassifier(objective='binary:logistic', use_label_encoder=False, eval_metric='logloss', random_state=42))
    ]
    param_grid = {
        'rf__n_estimators': [100, 200],
        'rf__max_depth': [10, None],
        'xgb__n_estimators': [100, 200],
        'xgb__max_depth': [3, 5],
        'xgb__learning_rate': [0.01, 0.1],
        'weights': [[1, 1], [1, 2], [2, 1]]
    }
    model, params, score = _optimize_model(VotingClassifier(estimators=estimators, voting='soft'),
                           param_grid, X_train, y_train, "Voting Classifier")
    return model, params, score

def optimize_stacking(X_train, y_train):
    # Expanded grid for Stacking Classifier
    estimators = [
        ('rf', RandomForestClassifier(random_state=42, class_weight='balanced')),
        ('xgb', xgb.XGBClassifier(objective='binary:logistic', use_label_encoder=False, eval_metric='logloss', random_state=42))
    ]
    final_estimator = LogisticRegression(class_weight='balanced', random_state=42, max_iter=1000)
    param_grid = {
        'rf__n_estimators': [100, 200],
        'rf__max_depth': [10, None],
        'xgb__n_estimators': [100, 200],
        'xgb__max_depth': [3, 5],
        'final_estimator__C': [0.1, 1.0, 10.0],
        'passthrough': [True, False],
        'stack_method': ['auto', 'predict_proba']
    }
    model, params, score = _optimize_model(StackingClassifier(estimators=estimators, final_estimator=final_estimator, cv=3),
                           param_grid, X_train, y_train, "Stacking Classifier")
    return model, params, score


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
        if model_name == 'naive_bayes': model = GaussianNB()
        elif model_name == 'xgboost': model = xgb.XGBClassifier(objective='binary:logistic', use_label_encoder=False, eval_metric='logloss', random_state=42)
        # elif model_name == 'svm': model = SVC(probability=True, class_weight='balanced', random_state=42)
        elif model_name == 'random_forest': model = RandomForestClassifier(class_weight='balanced', random_state=42)
        elif model_name == 'logistic_regression': model = LogisticRegression(class_weight='balanced', random_state=42, max_iter=1000)
        elif model_name == 'catboost': model = CatBoostClassifier(verbose=0, auto_class_weights='Balanced', random_state=42)
        elif model_name == 'lightgbm': model = lgb.LGBMClassifier(class_weight='balanced', verbose=-1, random_state=42)
        elif model_name == 'gradient_boosting': model = GradientBoostingClassifier(random_state=42)
        elif model_name == 'decision_tree': model = DecisionTreeClassifier(class_weight='balanced', random_state=42)
        # Voting and Stacking require base estimators, handled separately or need specific non-optimized setup
        elif model_name == 'voting':
            estimators = [
                ('rf', RandomForestClassifier(n_estimators=50, random_state=42, class_weight='balanced')),
                ('xgb', xgb.XGBClassifier(objective='binary:logistic', use_label_encoder=False, eval_metric='logloss', n_estimators=50, random_state=42))
            ]
            model = VotingClassifier(estimators=estimators, voting='soft')
        elif model_name == 'stacking':
            estimators = [
                ('rf', RandomForestClassifier(n_estimators=50, random_state=42, class_weight='balanced')),
                ('xgb', xgb.XGBClassifier(objective='binary:logistic', use_label_encoder=False, eval_metric='logloss', n_estimators=50, random_state=42))
            ]
            final_estimator = LogisticRegression(class_weight='balanced', random_state=42, max_iter=1000)
            model = StackingClassifier(estimators=estimators, final_estimator=final_estimator, cv=3) # cv=3 for speed
        elif model_name == 'mlp':
            if len(y_train) < 20: # Threshold: validation_fraction (0.1) * len(y_train) must be >= 2 (n_classes)
                logging.warning(f"MLP (non-optimized): Training data size ({len(y_train)}) is too small for early stopping with validation_fraction=0.1. Disabling early stopping for this run.")
                model = MLPClassifier(hidden_layer_sizes=(50,), max_iter=2000, early_stopping=False, random_state=42)
            else:
                model = MLPClassifier(hidden_layer_sizes=(50,), max_iter=2000, early_stopping=True, validation_fraction=0.1, n_iter_no_change=10, random_state=42) # Improved MLP
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


def analyze_project(project_name, level, resampling_strategy=None, n_folds=5, optimize=False, selected_features_config=None, methods_to_run=None, cli_args=None, progress=None, project_task_id=None):
    logging.info(f"Starting analysis for project: {project_name}, level: {level}, resampling: {resampling_strategy}, optimize: {optimize}")

    df = load_project_data(project_name, level)
    if df is None:
        logging.error(f"Failed to load data for {project_name} at level {level}. Skipping analysis.")
        return None

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

    results_base_dir = get_results_dir(level)
    project_results_dir = results_base_dir / project_name

    # Subdirectory for resampling strategy
    # Use the string name of the strategy (e.g., 'none', 'smote') for the directory name
    analysis_subdir_name = f"analysis_{resampling_strategy if resampling_strategy is not None else 'none'}"
    output_dir = project_results_dir / analysis_subdir_name
    output_dir.mkdir(parents=True, exist_ok=True)

    kfold = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    # Ensure the 'resampling' key in the results dict uses the path name ('none' or method name)
    results = {
        'project': project_name,
        'level': level,
        'resampling': resampling_strategy if resampling_strategy is not None else 'none'
    }

    # --- Dynamic Feature Selection Step ---
    if cli_args and cli_args.select_feature and selected_features_config is None:
        logging.info(f"Applying dynamic feature selection using method: {cli_args.select_feature} for project {project_name}, level {level}")

        k_fs = cli_args.k_features
        if k_fs is None:
            if X.shape[1] > 0: # Check if X has columns
                k_fs = max(1, X.shape[1] // 2)
            else:
                k_fs = 0 # No features to select from
        if X.shape[1] > 0: # Ensure k_fs is not more than available features only if X has features
            k_fs = min(k_fs, X.shape[1])
        else:
            k_fs = 0


        selected_features_list = []
        dynamic_fs_output_dir = output_dir / "dynamic_feature_selection_info"
        dynamic_fs_output_dir.mkdir(parents=True, exist_ok=True)

        if X.empty or X.shape[1] == 0:
            logging.warning(f"No features available in X for dynamic selection (Project: {project_name}, Level: {level}). Skipping dynamic feature selection.")
        elif len(y.unique()) < 2:
            logging.warning(f"Target variable 'y' has less than 2 unique classes for dynamic feature selection (Project: {project_name}, Level: {level}). Skipping dynamic feature selection.")
        else:
            if cli_args.select_feature == 'combine':
                all_methods_for_combine = ['variance', 'chi2', 'rfe', 'lasso', 'rf', 'mi']
                all_results_combine = {}
                for method_name_fs in all_methods_for_combine:
                    logging.info(f"  Running method '{method_name_fs}' as part of 'combine' strategy...")
                    try:
                        # k=X.shape[1] to get all scores for ranking, then combine and pick top k_fs
                        _, temp_selected_feats, temp_importance_scores = fs.select_features(X.copy(), y.copy(), method=method_name_fs, k=X.shape[1])
                        if temp_selected_feats:
                            all_results_combine[method_name_fs] = (None, temp_selected_feats, temp_importance_scores)
                            fs.analyze_feature_selection_results(X.copy(), temp_selected_feats, temp_importance_scores, f"combine_step_{method_name_fs}", dynamic_fs_output_dir)
                    except Exception as e_fs_combine_step:
                        logging.error(f"  Error running method '{method_name_fs}' for 'combine' strategy: {e_fs_combine_step}")

                if all_results_combine:
                    combined_ranked_features = fs.combine_feature_importance(all_results_combine)
                    selected_features_list = [feat for feat, score in combined_ranked_features[:k_fs]]

                    with open(dynamic_fs_output_dir / "combine_ranked_features.txt", 'w') as f_comb:
                        f_comb.write(f"Combined Feature Ranking (top {k_fs} selected):\n")
                        for feat, score in combined_ranked_features:
                            f_comb.write(f"{feat}: {score:.4f}{' (selected)' if feat in selected_features_list else ''}\n")
                    logging.info(f"  'combine' strategy selected {len(selected_features_list)} features: {selected_features_list}")
                else:
                    logging.warning("  'combine' strategy did not yield any results. Using all features from prepare_features.")
                    selected_features_list = X.columns.tolist() if not X.empty else []
            else: # Specific method
                try:
                    _, selected_features_list, importance_scores = fs.select_features(X.copy(), y.copy(), method=cli_args.select_feature, k=k_fs)
                    fs.analyze_feature_selection_results(X.copy(), selected_features_list, importance_scores, f"dynamic_{cli_args.select_feature}", dynamic_fs_output_dir)
                    logging.info(f"  Method '{cli_args.select_feature}' selected {len(selected_features_list)} features: {selected_features_list}")
                except Exception as e_fs_single:
                    logging.error(f"  Error during dynamic feature selection with method '{cli_args.select_feature}': {e_fs_single}. Using all features from prepare_features.")
                    selected_features_list = X.columns.tolist() if not X.empty else []

            if not selected_features_list and not X.empty:
                logging.warning(f"  Dynamic feature selection method '{cli_args.select_feature}' returned no features. Using all {len(X.columns)} features from prepare_features.")
                selected_features_list = X.columns.tolist()

            if selected_features_list:
                valid_selected_features = [col for col in selected_features_list if col in X.columns]
                if len(valid_selected_features) != len(selected_features_list):
                    logging.warning(f"  Some dynamically selected features were not found in X. Original: {selected_features_list}, Valid: {valid_selected_features}")

                if not valid_selected_features and not X.empty:
                    logging.warning(f"  No valid features remained after dynamic selection. Using all {len(X.columns)} features from before dynamic selection.")
                    # X remains unchanged if valid_selected_features is empty but X was not
                elif valid_selected_features:
                    X = X[valid_selected_features]
                    logging.info(f"  X updated to {X.shape[1]} features after dynamic selection.")
                # If valid_selected_features is empty AND X was already empty or became empty, X is now empty.
            else: # selected_features_list was empty
                logging.warning("  Dynamic feature selection resulted in an empty list of features. X remains unchanged or is empty.")

        # Save the list of features *actually used* for this run
        final_features_used_path = output_dir / "final_features_used_for_analysis.txt"
        with open(final_features_used_path, 'w') as f_final_feats:
            f_final_feats.write(f"Features used for analysis (Project: {project_name}, Level: {level}, Resampling: {resampling_strategy}, Dynamic FS: {cli_args.select_feature if cli_args else 'None'}, k: {cli_args.k_features if cli_args and cli_args.select_feature else 'N/A'}):\n")
            if not X.empty:
                for feat_name in X.columns:
                    f_final_feats.write(f"- {feat_name}\n")
            else:
                f_final_feats.write("No features were used (X is empty).\n")
        logging.info(f"List of final features used for this analysis run saved to {final_features_used_path}")

    # If after feature selection, X is empty, we should skip further processing for this project/strategy.
    if X.empty:
        logging.warning(f"No features remaining after dynamic selection for project {project_name}, level {level}, resampling {resampling_strategy}. Skipping model training for this configuration.")
        # Return a minimal result structure or None, indicating skipped analysis
        results['error'] = "No features remaining after dynamic selection"
        # Populate with zeroed metrics for all models that would have run
        for model_name_key in classifier_funcs_to_run.keys():
            results[model_name_key] = {
                'accuracy': 0, 'precision_0': 0, 'recall_0': 0, 'f1_0': 0,
                'precision_1': 0, 'recall_1': 0, 'f1_1': 0, 'auc': 0,
                'error': 'Skipped due to no features after dynamic selection'
            }
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
        # Ensure methods_to_run is a list
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
    plot_feature_correlations(X, output_dir, project_name, level)

    # --- New Parallel Task Preparation ---
    process_args = []
    # Loop over folds first to prepare data for each fold
    for fold, (train_idx, test_idx) in enumerate(kfold.split(X, y)):
        X_train_fold, X_test_fold = X.iloc[train_idx], X.iloc[test_idx]
        y_train_fold, y_test_fold = y.iloc[train_idx], y.iloc[test_idx]

        # Apply resampling to this fold's training data
        X_train_resampled, y_train_resampled = X_train_fold, y_train_fold
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
    # Group raw fold results by model name
    model_fold_results = {model_name: [] for model_name in classifier_funcs_to_run.keys()}
    for result_tuple in fold_results_list:
        if result_tuple:
            model_name, _, metrics = result_tuple  # fold_num is in metrics dict
            model_fold_results[model_name].append(metrics)

    # Process aggregated results for each model
    for model_name, fold_metrics_list in model_fold_results.items():
        if not fold_metrics_list:
            logging.error(f"No metrics collected for any fold for model {model_name}. Skipping aggregation.")
            results[model_name] = {'error': 'No fold metrics collected'}
            continue

        # --- NEW OPTIMIZATION RESULTS SAVING LOGIC ---
        if optimize and fold_metrics_list and any('best_cv_score' in d for d in fold_metrics_list):
            logging.info(f"Aggregating and saving optimization results for model: {model_name}")

            # 1. Prepare data for all folds that have optimization results
            all_results_to_log = []
            for fold_data in fold_metrics_list:
                if 'best_cv_score' not in fold_data or fold_data.get('best_params') is None:
                    continue  # Skip if a fold failed before or during optimization step

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

                # 2. Save all fold results to the _all_optimizations.json file
                combined_file_path = optimization_dir / f"{project_name}_{level}_{resampling_name}_all_optimizations.json"
                try:
                    with open(combined_file_path, 'w', encoding='utf-8') as f:
                        json.dump(convert_numpy_to_list_recursive(all_results_to_log), f, indent=2, ensure_ascii=False)
                    logging.info(f"All fold optimization results for '{model_name}' saved to {combined_file_path}")
                except Exception as e:
                    logging.error(f"Error saving all fold optimization results to {combined_file_path}: {e}")

                # 3. Find the best result based on test_performance.f1_score and save it
                best_fold_result_data = max(all_results_to_log, key=lambda x: x.get('test_performance', {}).get('f1_score', 0))
                single_file_path = optimization_dir / f"{project_name}_{level}_{resampling_name}_{model_name}_optimization.json"
                try:
                    with open(single_file_path, 'w', encoding='utf-8') as f:
                        json.dump(convert_numpy_to_list_recursive(best_fold_result_data), f, indent=2, ensure_ascii=False)
                    logging.info(f"Best optimization result for '{model_name}' saved to {single_file_path}")
                except Exception as e:
                    logging.error(f"Error saving single best optimization result to {single_file_path}: {e}")
        # --- END OF NEW LOGIC ---

        # --- Average Metrics ---
        metrics_to_average_list = []
        for metrics_dict in fold_metrics_list:
            if 'error' not in metrics_dict:
                avg_candidate_metrics = {
                    k: v for k, v in metrics_dict.items()
                    if k not in ['feature_importance', 'fpr', 'tpr', 'fold', 'error', 'y_test_fold', 'y_prob_fold']
                }
                numeric_metrics_for_avg = {k: v for k, v in avg_candidate_metrics.items() if isinstance(v, (int, float, np.number))}
                if numeric_metrics_for_avg:
                    metrics_to_average_list.append(numeric_metrics_for_avg)

        if not metrics_to_average_list:
            logging.error(f"No valid metrics to average for model {model_name}.")
            avg_metrics = {'error': "No valid fold metrics to average"}
        else:
            avg_metrics_df = pd.DataFrame(metrics_to_average_list)
            numeric_cols = avg_metrics_df.select_dtypes(include=np.number).columns
            avg_metrics = avg_metrics_df[numeric_cols].mean().to_dict()

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
    # The `results` dict already contains the correct resampling name for the plot title
    plot_scores_barchart(results, 'f1_1', 'F1 Scores (Class 1)', f'{project_name}_{level}_f1_scores', project_name, level, output_dir)
    plot_scores_barchart(results, 'accuracy', 'Accuracy Scores', f'{project_name}_{level}_accuracy_scores', project_name, level, output_dir)

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

def generate_markdown_tables(levels_to_process, project_name_filter=None):
    """Generate markdown tables from existing analysis results."""
    logging.info(f"Generating markdown tables for levels: {levels_to_process}")

    for level in levels_to_process:
        logging.info(f"Processing level: {level}")
        results_dir = get_results_dir(level)
        summary_dir = results_dir / "_summary"

        if not summary_dir.exists():
            logging.warning(f"Summary directory not found for level {level}: {summary_dir}")
            continue

        # Collect all CSV files from all resampling strategy subdirectories
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
    best_per_project_headers = ['Project', 'Resampling Method', 'ML Algorithm', 'F1-Score (Bug)', 'Accuracy', 'Precision (Bug)', 'Recall (Bug)', 'AUC']

    for _, row in best_per_project.iterrows():
        best_per_project_table_data.append([
            row.get('project', 'N/A'),
            row.get('resampling', 'N/A'),
            row.get('model', 'N/A'),
            format_metric(row.get('f1_1')),
            format_metric(row.get('accuracy')),
            format_metric(row.get('precision_1')),
            format_metric(row.get('recall_1')),
            format_metric(row.get('auc'))
        ])

    # Get best result for each ML algorithm based on F1-Score
    best_per_algorithm_table_data = []
    best_per_algorithm_headers = ['ML Algorithm', 'Project', 'Resampling Method', 'F1-Score (Bug)', 'Accuracy', 'Precision (Bug)', 'Recall (Bug)', 'AUC']
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
                format_metric(row.get('auc'))
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
        best_headers = ['Project', 'Resampling Method', 'ML Algorithm', 'Accuracy', 'Precision (Bug)', 'Recall (Bug)', 'F1-Score (Bug)', 'AUC']
        best_table_data = [[
            project_name,
            best_result.get('resampling', 'N/A'),
            best_result.get('model', 'N/A'),
            format_metric(best_result.get('accuracy')),
            format_metric(best_result.get('precision_1')),
            format_metric(best_result.get('recall_1')),
            format_metric(best_result.get('f1_1')),
            format_metric(best_result.get('auc'))
        ]]

        # Prepare table data for all resampling methods - show best model for each resampling method
        all_resampling_headers = ['Resampling Method', 'Best ML Algorithm', 'Accuracy', 'Precision (Bug)', 'Recall (Bug)', 'F1-Score (Bug)', 'AUC']
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
                format_metric(best_for_resampling.get('auc'))
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
    parser = argparse.ArgumentParser(description='Run bug prediction analysis for different granularity levels.')

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

    # Feature Selection Arguments
    parser.add_argument('--select-feature', type=str, default=None,
                        choices=['variance', 'chi2', 'rfe', 'lasso', 'rf', 'mi', 'combine'],
                        help='Dynamic feature selection method to use.')
    parser.add_argument('--k-features', type=int, default=None,
                        help='Number of top features to select for feature selection methods.')

    # Reporting and Utility Arguments
    parser.add_argument('--stats_only', action='store_true',
                        help='Generate and print dataset statistics for all levels (if --level is not specified).')
    parser.add_argument('--generate-tables', action='store_true',
                        help='Generate all summary markdown tables for the specified level.')
    parser.add_argument('--generate-summary-plots', action='store_true',
                        help='Generate summary bar charts from existing results.md files.')
    parser.add_argument('--regenerate-figures', action='store_true',
                        help='Regenerate all ROC and metric figures from existing JSON results.')
    parser.add_argument('--friedman-test', action='store_true', help='Run Friedman test on results.')
    parser.add_argument('--nemenyi-test', action='store_true', help='Run Nemenyi post-hoc test on results.')
    parser.add_argument('--metadata', action='store_true',
                        help='Generate a metadata.md file in a metadata directory.')

    parser.add_argument('--exclude-go-metrics', action='store_true',
                        help='Exclude Go-specific metrics from the analysis.')

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

    cli_args = parser.parse_args()

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
    is_statistical_test = cli_args.friedman_test or cli_args.nemenyi_test
    is_utility_mode = cli_args.regenerate_figures or cli_args.generate_tables or cli_args.stats_only or is_statistical_test or cli_args.important_features or cli_args.metadata or cli_args.best_features or cli_args.generate_summary_plots

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

    if cli_args.generate_tables:
        levels_to_process_results = [cli_args.level] if cli_args.level else ALL_LEVELS
        project_filter = cli_args.project if cli_args.project != 'all' else None
        generate_markdown_tables(levels_to_process_results, project_filter)
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
                    project_task_id=project_task_id
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
                        'resampling': res_dict.get('resampling') # This is set by analyze_project
                    }
                    for model_name, metrics_val in res_dict.items():
                        if model_name not in ['project', 'level', 'resampling']:
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

                summary_dir = get_results_dir(cli_args.level) / "_summary" / path_strategy_name # Use path_strategy_name for dir
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

    for strategy_name in resampling_strategies_to_run:
        actual_strategy_name = strategy_name if strategy_name is not None else 'none'
        logging.info(f"-- Regenerating figures for resampling strategy: {actual_strategy_name} --")
        summary_dir_for_strategy = get_results_dir(cli_args.level) / "_summary" / actual_strategy_name

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
        headers = ['Resampling Method', 'Accuracy', 'Precision (Bug)', 'Recall (Bug)', 'F1-Score (Bug)', 'AUC']
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
                    format_metric(result.get('auc'))
                ])
            else:
                table_data.append([
                    resampling_method,
                    'N/A', 'N/A', 'N/A', 'N/A', 'N/A'
                ])

        # Sort by F1-Score (Bug) in descending order
        table_data.sort(key=lambda x: float(x[4]) if x[4] != 'N/A' else -1, reverse=True)

        detailed_section += tabulate.tabulate(table_data, headers=headers, tablefmt='pipe') + "\n\n"

    # Add summary comparison table - best result for each algorithm
    detailed_section += "## Algorithm Performance Summary\n\n"
    detailed_section += "*Best result for each ML algorithm across all resampling methods*\n\n"

    summary_headers = ['ML Algorithm', 'Best Resampling Method', 'Accuracy', 'Precision (Bug)', 'Recall (Bug)', 'F1-Score (Bug)', 'AUC']
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
                    format_metric(best_result.get('auc'))
                ])
            else:
                # All f1_1 values are NaN for this algorithm
                summary_table_data.append([
                    algorithm.replace('_', ' ').title(),
                    'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A'
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
        if model_name in ['project', 'level', 'resampling', 'error']:
            continue

        if isinstance(metrics, dict):
            print(f"\n--- Model: {model_name.replace('_', ' ').title()} ---")

            table_data = []
            headers = ["Metric", "Value"]

            # Define the order of metrics to print
            metrics_to_print = [
                ('accuracy', 'Accuracy'),
                ('f1_1', 'F1 Score (Bug)'),
                ('precision_1', 'Precision (Bug)'),
                ('recall_1', 'Recall (Bug)'),
                ('auc', 'AUC'),
                ('f1_0', 'F1 Score (Clean)'),
                ('precision_0', 'Precision (Clean)'),
                ('recall_0', 'Recall (Clean)'),
            ]

            for key, name in metrics_to_print:
                value = metrics.get(key)
                if value is not None:
                    table_data.append([name, f"{value:.4f}"])

            if 'error' in metrics:
                 table_data.append(["Error", metrics['error']])

            print(tabulate.tabulate(table_data, headers=headers, tablefmt="grid"))

    print("\n" + "="*60)


if __name__ == "__main__":
    main()
