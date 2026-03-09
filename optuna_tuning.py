"""
Optuna-based Hyperparameter Tuning Module

This module implements a unified hyperparameter optimization protocol using Optuna-TPE
for all machine learning algorithms in the bug prediction study.

Protocol:
- Optimizer: Optuna with TPE (Tree-structured Parzen Estimator) sampler
- Budget: 100 trials per model (fixed for fairness and reproducibility)
- Seed: 42 (fixed everywhere for reproducibility)
- Primary metric: MCC (Matthews Correlation Coefficient) on inner validation
- Early stopping: Enabled for boosting models (XGBoost, LightGBM, CatBoost)

Reference:
- Akiba et al. (2019). Optuna: A Next-generation Hyperparameter Optimization Framework.
  KDD 2019. https://dl.acm.org/doi/10.1145/3292500.3330701
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple, List, Callable, Union
from dataclasses import dataclass, field
from pathlib import Path
import json
import datetime
import warnings

import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner

from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import matthews_corrcoef, make_scorer
from sklearn.base import clone

# ML Models
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import (
    RandomForestClassifier, 
    GradientBoostingClassifier,
    VotingClassifier,
    StackingClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier

import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier

# Imbalanced learning
from imblearn.pipeline import Pipeline as ImbPipeline

# Suppress Optuna's verbose logging
optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings('ignore', category=UserWarning, module='lightgbm')
warnings.filterwarnings('ignore', category=UserWarning, module='xgboost')


# =============================================================================
# Constants and Configuration
# =============================================================================

RANDOM_SEED = 42
N_TRIALS = 100
EARLY_STOPPING_ROUNDS = 50
MCC_SCORER = make_scorer(matthews_corrcoef)


@dataclass
class TuningConfig:
    """Configuration for hyperparameter tuning."""
    n_trials: int = N_TRIALS
    random_seed: int = RANDOM_SEED
    early_stopping_rounds: int = EARLY_STOPPING_ROUNDS
    optimization_metric: str = 'mcc'
    timeout: Optional[int] = None  # seconds, None = no timeout
    n_jobs: int = 1  # Sequential for reproducibility
    show_progress_bar: bool = False


@dataclass 
class TuningResult:
    """Result of hyperparameter tuning for a single model."""
    model_name: str
    best_params: Dict[str, Any]
    best_score: float
    n_trials: int
    study_name: str
    optimization_history: List[float] = field(default_factory=list)
    best_trial_number: int = 0
    tuning_time_seconds: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'model_name': self.model_name,
            'best_params': self.best_params,
            'best_score': self.best_score,
            'n_trials': self.n_trials,
            'study_name': self.study_name,
            'best_trial_number': self.best_trial_number,
            'tuning_time_seconds': self.tuning_time_seconds
        }


# =============================================================================
# Adaptive Inner CV
# =============================================================================

class AdaptiveInnerCV:
    """
    Adaptive inner cross-validation for hyperparameter tuning.
    
    Implements minority-class-aware fold selection:
    - k = min(5, n_minority) for number of folds
    - Falls back to single temporal validation split when k < 3
    - Respects temporal order (no future data leakage)
    
    Parameters
    ----------
    groups : array-like, optional
        Group labels (e.g., commit SHA) for group-aware splitting
    timestamps : array-like, optional
        Timestamps for temporal ordering
    min_folds : int, default=3
        Minimum number of folds required for k-fold CV; below this uses single split
    """
    
    def __init__(self, groups=None, timestamps=None, min_folds=3):
        self.groups = groups
        self.timestamps = timestamps
        self.min_folds = min_folds
        self._n_splits = None
        self._use_single_split = False
    
    def configure(self, y: np.ndarray) -> 'AdaptiveInnerCV':
        """
        Configure the CV strategy based on minority class count.
        
        Parameters
        ----------
        y : array-like
            Target labels
            
        Returns
        -------
        self : configured instance
        """
        y_array = np.asarray(y)
        n_minority = min(np.sum(y_array == 0), np.sum(y_array == 1))
        
        # k = min(5, n_minority)
        k = min(5, int(n_minority))
        
        if k < self.min_folds:
            self._use_single_split = True
            self._n_splits = 1
            logging.info(
                f"AdaptiveInnerCV: n_minority={n_minority}, k={k} < {self.min_folds}, "
                f"using single temporal validation split"
            )
        else:
            self._use_single_split = False
            self._n_splits = k
            logging.info(
                f"AdaptiveInnerCV: n_minority={n_minority}, using {k}-fold temporal CV"
            )
        
        return self
    
    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        """Get number of splits."""
        return self._n_splits if self._n_splits else 3
    
    def split(self, X, y=None, groups=None):
        """
        Generate train/validation indices.
        
        Yields
        ------
        train_indices, val_indices : arrays
        """
        n_samples = len(X) if hasattr(X, '__len__') else X.shape[0]
        
        if self._use_single_split:
            # Single temporal split: 70% train, 30% validation
            split_idx = int(n_samples * 0.7)
            train_indices = np.arange(split_idx)
            val_indices = np.arange(split_idx, n_samples)
            yield train_indices, val_indices
            return
        
        # k-fold temporal CV
        if self.groups is not None and len(self.groups) == n_samples:
            yield from self._group_aware_split(X, y, groups)
        else:
            yield from self._simple_temporal_split(n_samples)
    
    def _group_aware_split(self, X, y, groups):
        """Group-aware temporal splitting (respects commit boundaries)."""
        groups_array = np.array(self.groups) if self.groups is not None else groups
        
        # Get unique groups in order
        unique_groups = []
        seen = set()
        for g in groups_array:
            if g not in seen:
                unique_groups.append(g)
                seen.add(g)
        
        n_groups = len(unique_groups)
        n_splits = self._n_splits or 3
        
        # Ensure we have enough groups
        if n_groups < n_splits + 1:
            n_splits = max(2, n_groups - 1)
        
        groups_per_fold = max(1, n_groups // (n_splits + 1))
        
        for fold in range(n_splits):
            train_end_idx = groups_per_fold * (fold + 1)
            val_end_idx = min(train_end_idx + groups_per_fold, n_groups)
            
            train_groups = set(unique_groups[:train_end_idx])
            val_groups = set(unique_groups[train_end_idx:val_end_idx])
            
            train_indices = np.where(np.isin(groups_array, list(train_groups)))[0]
            val_indices = np.where(np.isin(groups_array, list(val_groups)))[0]
            
            if len(train_indices) > 0 and len(val_indices) > 0:
                yield train_indices, val_indices
    
    def _simple_temporal_split(self, n_samples):
        """Simple sequential temporal splitting."""
        n_splits = self._n_splits or 3
        fold_size = n_samples // (n_splits + 1)
        
        for fold in range(n_splits):
            train_end = fold_size * (fold + 1)
            val_end = min(train_end + fold_size, n_samples)
            
            train_indices = np.arange(train_end)
            val_indices = np.arange(train_end, val_end)
            
            if len(train_indices) > 0 and len(val_indices) > 0:
                yield train_indices, val_indices


def get_adaptive_inner_cv(y_train: np.ndarray, groups=None, timestamps=None) -> AdaptiveInnerCV:
    """
    Factory function to create and configure an AdaptiveInnerCV instance.
    
    Implements the rule: k = min(5, n_minority), fallback to single split if k < 3.
    
    Parameters
    ----------
    y_train : array-like
        Training labels for minority class calculation
    groups : array-like, optional
        Group labels for group-aware splitting
    timestamps : array-like, optional
        Timestamps for temporal ordering
        
    Returns
    -------
    AdaptiveInnerCV : configured CV splitter
    """
    cv = AdaptiveInnerCV(groups=groups, timestamps=timestamps, min_folds=3)
    cv.configure(y_train)
    return cv


# =============================================================================
# Search Space Definitions
# =============================================================================

SEARCH_SPACES = {
    'xgboost': {
        'n_estimators': ('int', 200, 3000),
        'max_depth': ('int', 3, 12),
        'learning_rate': ('float_log', 0.01, 0.3),
        'subsample': ('float', 0.5, 1.0),
        'colsample_bytree': ('float', 0.5, 1.0),
        'min_child_weight': ('float_log', 1, 10),
        'reg_alpha': ('float_log', 1e-8, 10),
        'reg_lambda': ('float_log', 1e-8, 10),
        'gamma': ('float_log', 1e-8, 10),
    },
    'lightgbm': {
        'n_estimators': ('int', 200, 5000),
        'num_leaves': ('int', 16, 256),
        'max_depth': ('int', -1, 12),
        'learning_rate': ('float_log', 0.01, 0.3),
        'feature_fraction': ('float', 0.5, 1.0),
        'bagging_fraction': ('float', 0.5, 1.0),
        'bagging_freq': ('int', 0, 10),
        'min_data_in_leaf': ('int', 10, 200),
        'lambda_l1': ('float_log', 1e-8, 10),
        'lambda_l2': ('float_log', 1e-8, 10),
    },
    'catboost': {
        'iterations': ('int', 500, 5000),
        'depth': ('int', 4, 10),
        'learning_rate': ('float_log', 0.01, 0.3),
        'l2_leaf_reg': ('float_log', 1, 20),
        'random_strength': ('float', 0, 10),
        'bagging_temperature': ('float', 0, 1),
        'rsm': ('float', 0.5, 1.0),
    },
    'random_forest': {
        'n_estimators': ('int', 200, 3000),
        'max_depth': ('categorical', [None, 5, 10, 20, 40]),
        'min_samples_split': ('int', 2, 20),
        'min_samples_leaf': ('int', 1, 10),
        'max_features': ('categorical', ['sqrt', 'log2', 0.5]),
        'bootstrap': ('categorical', [True, False]),
    },
    'logistic_regression': {
        'C': ('float_log', 1e-4, 100),
        'penalty': ('categorical', ['l2', 'l1']),
        'class_weight': ('categorical', [None, 'balanced']),
    },
    'gradient_boosting': {
        'n_estimators': ('int', 100, 2000),
        'learning_rate': ('float_log', 0.01, 0.3),
        'max_depth': ('int', 2, 6),
        'subsample': ('float', 0.5, 1.0),
        'min_samples_leaf': ('int', 1, 20),
        'max_features': ('categorical', ['sqrt', 'log2', None]),
    },
    'decision_tree': {
        'criterion': ('categorical', ['gini', 'entropy', 'log_loss']),
        'max_depth': ('categorical', [None, 5, 10, 20, 40]),
        'min_samples_split': ('int', 2, 20),
        'min_samples_leaf': ('int', 1, 20),
        'max_features': ('categorical', ['sqrt', 'log2', None]),
    },
    'mlp': {
        'hidden_layer_sizes': ('categorical', [(64,), (128,), (256,), (128, 64), (256, 128)]),
        'alpha': ('float_log', 1e-6, 1e-2),
        'learning_rate_init': ('float_log', 1e-4, 1e-2),
        'batch_size': ('categorical', [64, 128, 256]),
        'activation': ('categorical', ['relu', 'tanh']),
    },
    'stacking_meta': {
        'passthrough': ('categorical', [True, False]),
        'final_estimator__C': ('float_log', 1e-4, 100),
    },
}


def suggest_params(trial: optuna.Trial, model_name: str) -> Dict[str, Any]:
    """
    Suggest hyperparameters for a given model using Optuna trial.
    
    Parameters
    ----------
    trial : optuna.Trial
        Optuna trial object
    model_name : str
        Name of the model (key in SEARCH_SPACES)
        
    Returns
    -------
    dict : suggested parameters
    """
    if model_name not in SEARCH_SPACES:
        raise ValueError(f"Unknown model: {model_name}")
    
    params = {}
    space = SEARCH_SPACES[model_name]
    
    for param_name, param_config in space.items():
        param_type = param_config[0]
        
        if param_type == 'int':
            params[param_name] = trial.suggest_int(param_name, param_config[1], param_config[2])
        elif param_type == 'float':
            params[param_name] = trial.suggest_float(param_name, param_config[1], param_config[2])
        elif param_type == 'float_log':
            params[param_name] = trial.suggest_float(param_name, param_config[1], param_config[2], log=True)
        elif param_type == 'categorical':
            params[param_name] = trial.suggest_categorical(param_name, param_config[1])
    
    return params


# =============================================================================
# Objective Functions
# =============================================================================

def create_xgboost_objective(
    X_train: np.ndarray, 
    y_train: np.ndarray,
    inner_cv: AdaptiveInnerCV,
    resampler=None,
    class_weight: str = None
) -> Callable:
    """
    Create XGBoost objective function with early stopping.
    
    Early stopping uses the validation fold from inner CV.
    """
    def objective(trial: optuna.Trial) -> float:
        params = suggest_params(trial, 'xgboost')
        
        # Fixed parameters
        params['objective'] = 'binary:logistic'
        params['eval_metric'] = 'logloss'
        params['random_state'] = RANDOM_SEED
        params['use_label_encoder'] = False
        params['early_stopping_rounds'] = EARLY_STOPPING_ROUNDS
        
        # Handle class weight
        if class_weight == 'balanced':
            n_pos = np.sum(y_train == 1)
            n_neg = np.sum(y_train == 0)
            params['scale_pos_weight'] = n_neg / n_pos if n_pos > 0 else 1
        
        scores = []
        for train_idx, val_idx in inner_cv.split(X_train, y_train):
            X_tr, X_val = X_train[train_idx], X_train[val_idx]
            y_tr, y_val = y_train[train_idx], y_train[val_idx]
            
            # Apply resampling only to training portion
            if resampler is not None:
                try:
                    X_tr, y_tr = resampler.fit_resample(X_tr, y_tr)
                except Exception:
                    pass  # Skip resampling if it fails
            
            model = xgb.XGBClassifier(**params)
            model.fit(
                X_tr, y_tr,
                eval_set=[(X_val, y_val)],
                verbose=False
            )
            
            y_pred = model.predict(X_val)
            mcc = matthews_corrcoef(y_val, y_pred)
            scores.append(mcc)
        
        return np.mean(scores) if scores else 0.0
    
    return objective


def create_lightgbm_objective(
    X_train: np.ndarray,
    y_train: np.ndarray,
    inner_cv: AdaptiveInnerCV,
    resampler=None,
    class_weight: str = None
) -> Callable:
    """
    Create LightGBM objective function with early stopping.
    """
    def objective(trial: optuna.Trial) -> float:
        params = suggest_params(trial, 'lightgbm')
        
        # Fixed parameters
        params['objective'] = 'binary'
        params['metric'] = 'binary_logloss'
        params['random_state'] = RANDOM_SEED
        params['verbose'] = -1
        params['force_col_wise'] = True
        
        # Handle class weight
        if class_weight == 'balanced':
            params['class_weight'] = 'balanced'
        
        # Early stopping callback
        callbacks = [lgb.early_stopping(stopping_rounds=EARLY_STOPPING_ROUNDS, verbose=False)]
        
        scores = []
        for train_idx, val_idx in inner_cv.split(X_train, y_train):
            X_tr, X_val = X_train[train_idx], X_train[val_idx]
            y_tr, y_val = y_train[train_idx], y_train[val_idx]
            
            if resampler is not None:
                try:
                    X_tr, y_tr = resampler.fit_resample(X_tr, y_tr)
                except Exception:
                    pass
            
            model = lgb.LGBMClassifier(**params)
            model.fit(
                X_tr, y_tr,
                eval_set=[(X_val, y_val)],
                callbacks=callbacks
            )
            
            y_pred = model.predict(X_val)
            mcc = matthews_corrcoef(y_val, y_pred)
            scores.append(mcc)
        
        return np.mean(scores) if scores else 0.0
    
    return objective


def create_catboost_objective(
    X_train: np.ndarray,
    y_train: np.ndarray,
    inner_cv: AdaptiveInnerCV,
    resampler=None,
    class_weight: str = None
) -> Callable:
    """
    Create CatBoost objective function with early stopping.
    """
    def objective(trial: optuna.Trial) -> float:
        params = suggest_params(trial, 'catboost')
        
        # Fixed parameters
        params['loss_function'] = 'Logloss'
        params['random_seed'] = RANDOM_SEED
        params['verbose'] = False
        params['od_type'] = 'Iter'
        params['od_wait'] = EARLY_STOPPING_ROUNDS
        
        # Handle class weight
        if class_weight == 'balanced':
            params['auto_class_weights'] = 'Balanced'
        
        scores = []
        for train_idx, val_idx in inner_cv.split(X_train, y_train):
            X_tr, X_val = X_train[train_idx], X_train[val_idx]
            y_tr, y_val = y_train[train_idx], y_train[val_idx]
            
            if resampler is not None:
                try:
                    X_tr, y_tr = resampler.fit_resample(X_tr, y_tr)
                except Exception:
                    pass
            
            model = CatBoostClassifier(**params)
            model.fit(
                X_tr, y_tr,
                eval_set=(X_val, y_val),
                verbose=False
            )
            
            y_pred = model.predict(X_val)
            mcc = matthews_corrcoef(y_val, y_pred)
            scores.append(mcc)
        
        return np.mean(scores) if scores else 0.0
    
    return objective


def create_sklearn_objective(
    model_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    inner_cv: AdaptiveInnerCV,
    resampler=None,
    class_weight: str = None,
    use_scaler: bool = False
) -> Callable:
    """
    Create objective function for sklearn models (RF, LR, GB, DT, MLP).
    """
    def objective(trial: optuna.Trial) -> float:
        params = suggest_params(trial, model_name)
        
        # Add fixed parameters and create model
        if model_name == 'random_forest':
            if class_weight == 'balanced':
                params['class_weight'] = 'balanced'
            params['random_state'] = RANDOM_SEED
            params['n_jobs'] = 1
            model = RandomForestClassifier(**params)
            
        elif model_name == 'logistic_regression':
            # Set solver based on penalty
            if params.get('penalty') == 'l1':
                params['solver'] = 'liblinear'
            else:
                params['solver'] = 'lbfgs'
            params['random_state'] = RANDOM_SEED
            params['max_iter'] = 2000
            model = LogisticRegression(**params)
            
        elif model_name == 'gradient_boosting':
            params['random_state'] = RANDOM_SEED
            model = GradientBoostingClassifier(**params)
            
        elif model_name == 'decision_tree':
            if class_weight == 'balanced':
                params['class_weight'] = 'balanced'
            params['random_state'] = RANDOM_SEED
            model = DecisionTreeClassifier(**params)
            
        elif model_name == 'mlp':
            params['random_state'] = RANDOM_SEED
            params['max_iter'] = 2000
            params['early_stopping'] = True
            params['n_iter_no_change'] = 20
            params['validation_fraction'] = 0.1
            model = MLPClassifier(**params)
        else:
            raise ValueError(f"Unknown sklearn model: {model_name}")
        
        scores = []
        for train_idx, val_idx in inner_cv.split(X_train, y_train):
            X_tr, X_val = X_train[train_idx], X_train[val_idx]
            y_tr, y_val = y_train[train_idx], y_train[val_idx]
            
            # Apply resampling
            if resampler is not None:
                try:
                    X_tr, y_tr = resampler.fit_resample(X_tr, y_tr)
                except Exception:
                    pass
            
            # Apply scaling if needed (MLP, LogisticRegression)
            if use_scaler:
                scaler = StandardScaler()
                X_tr = scaler.fit_transform(X_tr)
                X_val = scaler.transform(X_val)
            
            try:
                model_clone = clone(model)
                model_clone.fit(X_tr, y_tr)
                y_pred = model_clone.predict(X_val)
                mcc = matthews_corrcoef(y_val, y_pred)
                scores.append(mcc)
            except Exception as e:
                logging.warning(f"Model fitting failed: {e}")
                scores.append(0.0)
        
        return np.mean(scores) if scores else 0.0
    
    return objective


def create_stacking_meta_objective(
    X_train: np.ndarray,
    y_train: np.ndarray,
    inner_cv: AdaptiveInnerCV,
    base_estimators: List[Tuple[str, Any]],
    resampler=None,
    class_weight: str = None
) -> Callable:
    """
    Create objective function for StackingClassifier meta-learner tuning.
    
    Only tunes: passthrough and meta-learner's C parameter.
    Base estimators are fixed (already tuned).
    """
    def objective(trial: optuna.Trial) -> float:
        passthrough = trial.suggest_categorical('passthrough', [True, False])
        meta_C = trial.suggest_float('final_estimator__C', 1e-4, 100, log=True)
        
        # Create meta-learner
        meta_class_weight = 'balanced' if class_weight == 'balanced' else None
        final_estimator = LogisticRegression(
            C=meta_C, 
            class_weight=meta_class_weight,
            random_state=RANDOM_SEED, 
            max_iter=2000
        )
        
        scores = []
        for train_idx, val_idx in inner_cv.split(X_train, y_train):
            X_tr, X_val = X_train[train_idx], X_train[val_idx]
            y_tr, y_val = y_train[train_idx], y_train[val_idx]
            
            if resampler is not None:
                try:
                    X_tr, y_tr = resampler.fit_resample(X_tr, y_tr)
                except Exception:
                    pass
            
            # Clone base estimators for this fold
            fold_base_estimators = [(name, clone(est)) for name, est in base_estimators]
            
            model = StackingClassifier(
                estimators=fold_base_estimators,
                final_estimator=clone(final_estimator),
                passthrough=passthrough,
                cv=3,  # Internal CV for stacking
                n_jobs=1
            )
            
            try:
                model.fit(X_tr, y_tr)
                y_pred = model.predict(X_val)
                mcc = matthews_corrcoef(y_val, y_pred)
                scores.append(mcc)
            except Exception as e:
                logging.warning(f"Stacking fitting failed: {e}")
                scores.append(0.0)
        
        return np.mean(scores) if scores else 0.0
    
    return objective


# =============================================================================
# Main Tuner Class
# =============================================================================

class OptunaHyperparameterTuner:
    """
    Unified hyperparameter tuner using Optuna-TPE.
    
    Implements consistent tuning protocol:
    - TPE sampler with seed=42
    - 100 trials per model
    - MCC optimization metric
    - Early stopping for boosting models
    - Adaptive inner CV based on minority class count
    
    Parameters
    ----------
    config : TuningConfig
        Tuning configuration
    output_dir : Path, optional
        Directory to save tuning results
    """
    
    def __init__(self, config: TuningConfig = None, output_dir: Path = None):
        self.config = config or TuningConfig()
        self.output_dir = Path(output_dir) if output_dir else None
        self.results: Dict[str, TuningResult] = {}
        self._tuned_models: Dict[str, Any] = {}
        
    def tune_model(
        self,
        model_name: str,
        X_train: np.ndarray,
        y_train: np.ndarray,
        groups: np.ndarray = None,
        timestamps: np.ndarray = None,
        resampler=None,
        class_weight: str = None
    ) -> Tuple[Any, Dict[str, Any], float]:
        """
        Tune a single model using Optuna-TPE.
        
        Parameters
        ----------
        model_name : str
            Name of the model to tune
        X_train : array-like
            Training features
        y_train : array-like
            Training labels
        groups : array-like, optional
            Group labels for group-aware CV
        timestamps : array-like, optional
            Timestamps for temporal ordering
        resampler : object, optional
            Resampling method instance
        class_weight : str, optional
            'balanced' or None
            
        Returns
        -------
        model : fitted model with best parameters
        best_params : dict of best parameters
        best_score : float, best MCC score
        """
        import time
        start_time = time.time()
        
        # Convert to numpy if needed
        X_train = np.asarray(X_train)
        y_train = np.asarray(y_train)
        
        # Create adaptive inner CV
        inner_cv = get_adaptive_inner_cv(y_train, groups=groups, timestamps=timestamps)
        
        # Determine if scaling is needed
        use_scaler = model_name in ['mlp', 'logistic_regression']
        
        # Create objective function
        if model_name == 'xgboost':
            objective = create_xgboost_objective(
                X_train, y_train, inner_cv, resampler, class_weight
            )
        elif model_name == 'lightgbm':
            objective = create_lightgbm_objective(
                X_train, y_train, inner_cv, resampler, class_weight
            )
        elif model_name == 'catboost':
            objective = create_catboost_objective(
                X_train, y_train, inner_cv, resampler, class_weight
            )
        elif model_name in ['random_forest', 'logistic_regression', 'gradient_boosting', 
                            'decision_tree', 'mlp']:
            objective = create_sklearn_objective(
                model_name, X_train, y_train, inner_cv, resampler, class_weight, use_scaler
            )
        else:
            raise ValueError(f"Unknown model for tuning: {model_name}")
        
        # Create study with TPE sampler
        sampler = TPESampler(seed=self.config.random_seed)
        study_name = f"{model_name}_study"
        
        study = optuna.create_study(
            direction='maximize',  # Maximize MCC
            sampler=sampler,
            study_name=study_name
        )
        
        # Run optimization
        logging.info(f"Starting Optuna tuning for {model_name} with {self.config.n_trials} trials...")
        
        study.optimize(
            objective,
            n_trials=self.config.n_trials,
            timeout=self.config.timeout,
            n_jobs=self.config.n_jobs,
            show_progress_bar=self.config.show_progress_bar
        )
        
        tuning_time = time.time() - start_time
        
        # Get best parameters
        best_params = study.best_params
        best_score = study.best_value
        
        logging.info(f"Best MCC for {model_name}: {best_score:.4f}")
        logging.info(f"Best params: {best_params}")
        
        # Create and fit final model with best parameters
        model = self._create_model_with_params(model_name, best_params, class_weight)
        
        # Apply resampling to full training data if needed
        X_fit, y_fit = X_train, y_train
        if resampler is not None:
            try:
                X_fit, y_fit = resampler.fit_resample(X_train, y_train)
            except Exception:
                pass
        
        # Scale if needed
        if use_scaler:
            scaler = StandardScaler()
            X_fit = scaler.fit_transform(X_fit)
        
        # Fit final model
        model.fit(X_fit, y_fit)
        
        # Store result
        result = TuningResult(
            model_name=model_name,
            best_params=best_params,
            best_score=best_score,
            n_trials=len(study.trials),
            study_name=study_name,
            optimization_history=[t.value for t in study.trials if t.value is not None],
            best_trial_number=study.best_trial.number,
            tuning_time_seconds=tuning_time
        )
        self.results[model_name] = result
        self._tuned_models[model_name] = model
        
        # Save results if output_dir specified
        if self.output_dir:
            self._save_tuning_result(result)
        
        return model, best_params, best_score
    
    def _create_model_with_params(
        self, 
        model_name: str, 
        params: Dict[str, Any],
        class_weight: str = None
    ) -> Any:
        """Create a model instance with given parameters."""
        
        if model_name == 'xgboost':
            model_params = params.copy()
            model_params['objective'] = 'binary:logistic'
            model_params['eval_metric'] = 'logloss'
            model_params['random_state'] = RANDOM_SEED
            model_params['use_label_encoder'] = False
            return xgb.XGBClassifier(**model_params)
            
        elif model_name == 'lightgbm':
            model_params = params.copy()
            model_params['objective'] = 'binary'
            model_params['random_state'] = RANDOM_SEED
            model_params['verbose'] = -1
            if class_weight == 'balanced':
                model_params['class_weight'] = 'balanced'
            return lgb.LGBMClassifier(**model_params)
            
        elif model_name == 'catboost':
            model_params = params.copy()
            model_params['loss_function'] = 'Logloss'
            model_params['random_seed'] = RANDOM_SEED
            model_params['verbose'] = False
            if class_weight == 'balanced':
                model_params['auto_class_weights'] = 'Balanced'
            return CatBoostClassifier(**model_params)
            
        elif model_name == 'random_forest':
            model_params = params.copy()
            model_params['random_state'] = RANDOM_SEED
            model_params['n_jobs'] = 1
            if class_weight == 'balanced':
                model_params['class_weight'] = 'balanced'
            return RandomForestClassifier(**model_params)
            
        elif model_name == 'logistic_regression':
            model_params = params.copy()
            if model_params.get('penalty') == 'l1':
                model_params['solver'] = 'liblinear'
            else:
                model_params['solver'] = 'lbfgs'
            model_params['random_state'] = RANDOM_SEED
            model_params['max_iter'] = 2000
            return LogisticRegression(**model_params)
            
        elif model_name == 'gradient_boosting':
            model_params = params.copy()
            model_params['random_state'] = RANDOM_SEED
            return GradientBoostingClassifier(**model_params)
            
        elif model_name == 'decision_tree':
            model_params = params.copy()
            model_params['random_state'] = RANDOM_SEED
            if class_weight == 'balanced':
                model_params['class_weight'] = 'balanced'
            return DecisionTreeClassifier(**model_params)
            
        elif model_name == 'mlp':
            model_params = params.copy()
            model_params['random_state'] = RANDOM_SEED
            model_params['max_iter'] = 2000
            model_params['early_stopping'] = True
            model_params['n_iter_no_change'] = 20
            return MLPClassifier(**model_params)
        
        else:
            raise ValueError(f"Unknown model: {model_name}")
    
    def tune_stacking(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        base_estimators: List[Tuple[str, Any]],
        groups: np.ndarray = None,
        timestamps: np.ndarray = None,
        resampler=None,
        class_weight: str = None
    ) -> Tuple[StackingClassifier, Dict[str, Any], float]:
        """
        Tune StackingClassifier (meta-learner only).
        
        Base estimators should already be tuned.
        Only tunes: passthrough and meta-learner C.
        """
        import time
        start_time = time.time()
        
        X_train = np.asarray(X_train)
        y_train = np.asarray(y_train)
        
        inner_cv = get_adaptive_inner_cv(y_train, groups=groups, timestamps=timestamps)
        
        objective = create_stacking_meta_objective(
            X_train, y_train, inner_cv, base_estimators, resampler, class_weight
        )
        
        sampler = TPESampler(seed=self.config.random_seed)
        study = optuna.create_study(
            direction='maximize',
            sampler=sampler,
            study_name='stacking_meta_study'
        )
        
        logging.info(f"Starting Optuna tuning for Stacking meta-learner with {self.config.n_trials} trials...")
        
        study.optimize(
            objective,
            n_trials=self.config.n_trials,
            timeout=self.config.timeout,
            n_jobs=self.config.n_jobs,
            show_progress_bar=self.config.show_progress_bar
        )
        
        tuning_time = time.time() - start_time
        
        best_params = study.best_params
        best_score = study.best_value
        
        logging.info(f"Best MCC for Stacking: {best_score:.4f}")
        
        # Create final stacking model
        meta_class_weight = 'balanced' if class_weight == 'balanced' else None
        final_estimator = LogisticRegression(
            C=best_params['final_estimator__C'],
            class_weight=meta_class_weight,
            random_state=RANDOM_SEED,
            max_iter=2000
        )
        
        # Clone base estimators
        final_base_estimators = [(name, clone(est)) for name, est in base_estimators]
        
        model = StackingClassifier(
            estimators=final_base_estimators,
            final_estimator=final_estimator,
            passthrough=best_params['passthrough'],
            cv=3,
            n_jobs=1
        )
        
        # Fit on training data
        X_fit, y_fit = X_train, y_train
        if resampler is not None:
            try:
                X_fit, y_fit = resampler.fit_resample(X_train, y_train)
            except Exception:
                pass
        
        model.fit(X_fit, y_fit)
        
        # Store result
        result = TuningResult(
            model_name='stacking',
            best_params=best_params,
            best_score=best_score,
            n_trials=len(study.trials),
            study_name='stacking_meta_study',
            best_trial_number=study.best_trial.number,
            tuning_time_seconds=tuning_time
        )
        self.results['stacking'] = result
        self._tuned_models['stacking'] = model
        
        if self.output_dir:
            self._save_tuning_result(result)
        
        return model, best_params, best_score
    
    def get_voting_classifier(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        tuned_estimators: Dict[str, Any],
        resampler=None
    ) -> VotingClassifier:
        """
        Create VotingClassifier with tuned base estimators.
        
        NOT tuned - uses fixed equal weights and soft voting.
        Base estimators: [LR, RF, XGB, LGBM, CatBoost]
        
        Parameters
        ----------
        tuned_estimators : dict
            Dictionary mapping model names to tuned model instances
        """
        # Define base estimator names (in order)
        base_names = ['logistic_regression', 'random_forest', 'xgboost', 'lightgbm', 'catboost']
        
        estimators = []
        for name in base_names:
            if name in tuned_estimators:
                estimators.append((name, clone(tuned_estimators[name])))
            else:
                logging.warning(f"Tuned {name} not found for VotingClassifier, skipping")
        
        if len(estimators) < 2:
            raise ValueError("VotingClassifier requires at least 2 base estimators")
        
        model = VotingClassifier(
            estimators=estimators,
            voting='soft',  # Use probabilities
            weights=None,   # Equal weights
            n_jobs=1
        )
        
        # Fit on training data
        X_fit, y_fit = np.asarray(X_train), np.asarray(y_train)
        if resampler is not None:
            try:
                X_fit, y_fit = resampler.fit_resample(X_fit, y_fit)
            except Exception:
                pass
        
        model.fit(X_fit, y_fit)
        
        self._tuned_models['voting'] = model
        
        return model
    
    def get_naive_bayes(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        resampler=None
    ) -> GaussianNB:
        """
        Create GaussianNB with default parameters.
        
        NOT tuned - uses sklearn defaults.
        """
        model = GaussianNB()
        
        X_fit, y_fit = np.asarray(X_train), np.asarray(y_train)
        if resampler is not None:
            try:
                X_fit, y_fit = resampler.fit_resample(X_fit, y_fit)
            except Exception:
                pass
        
        model.fit(X_fit, y_fit)
        
        self._tuned_models['naive_bayes'] = model
        
        return model
    
    def _save_tuning_result(self, result: TuningResult):
        """Save tuning result to JSON file."""
        if self.output_dir is None:
            return
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        filepath = self.output_dir / f"{result.model_name}_tuning_result.json"
        with open(filepath, 'w') as f:
            json.dump(result.to_dict(), f, indent=2, default=str)
        
        logging.info(f"Saved tuning result to {filepath}")
    
    def save_all_results(self, filename: str = "all_tuning_results.json"):
        """Save all tuning results to a single JSON file."""
        if self.output_dir is None:
            return
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        all_results = {name: result.to_dict() for name, result in self.results.items()}
        
        filepath = self.output_dir / filename
        with open(filepath, 'w') as f:
            json.dump(all_results, f, indent=2, default=str)
        
        logging.info(f"Saved all tuning results to {filepath}")
    
    def save_best_params_csv(self, filename: str = "best_params.csv"):
        """Save best parameters to CSV file."""
        if self.output_dir is None:
            return
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        rows = []
        for model_name, result in self.results.items():
            for param_name, param_value in result.best_params.items():
                rows.append({
                    'model': model_name,
                    'parameter': param_name,
                    'value': param_value,
                    'best_mcc': result.best_score,
                    'n_trials': result.n_trials,
                    'tuning_time_seconds': result.tuning_time_seconds
                })
        
        df = pd.DataFrame(rows)
        filepath = self.output_dir / filename
        df.to_csv(filepath, index=False)
        
        logging.info(f"Saved best params to {filepath}")


# =============================================================================
# Convenience Functions for Integration with analiz.py
# =============================================================================

def tune_all_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    groups: np.ndarray = None,
    timestamps: np.ndarray = None,
    resampler=None,
    class_weight: str = None,
    output_dir: Path = None,
    config: TuningConfig = None
) -> Dict[str, Tuple[Any, Dict[str, Any], float]]:
    """
    Tune all models and return dictionary of (model, params, score).
    
    Tuned models: xgboost, lightgbm, catboost, random_forest, 
                  logistic_regression, gradient_boosting, decision_tree, mlp
    
    Not tuned: naive_bayes, voting, stacking (requires tuned base estimators)
    """
    tuner = OptunaHyperparameterTuner(config=config, output_dir=output_dir)
    
    results = {}
    
    # Tune individual models
    tuned_model_names = [
        'xgboost', 'lightgbm', 'catboost', 'random_forest',
        'logistic_regression', 'gradient_boosting', 'decision_tree', 'mlp'
    ]
    
    for model_name in tuned_model_names:
        try:
            model, params, score = tuner.tune_model(
                model_name, X_train, y_train, groups, timestamps, resampler, class_weight
            )
            results[model_name] = (model, params, score)
        except Exception as e:
            logging.error(f"Failed to tune {model_name}: {e}")
    
    # Create Naive Bayes (not tuned)
    nb_model = tuner.get_naive_bayes(X_train, y_train, resampler)
    results['naive_bayes'] = (nb_model, {}, None)
    
    # Create Voting Classifier (not tuned, uses tuned base estimators)
    tuned_estimators = {name: model for name, (model, _, _) in results.items()
                       if name in ['logistic_regression', 'random_forest', 'xgboost', 
                                   'lightgbm', 'catboost']}
    if len(tuned_estimators) >= 2:
        voting_model = tuner.get_voting_classifier(X_train, y_train, tuned_estimators, resampler)
        results['voting'] = (voting_model, {'voting': 'soft', 'weights': 'equal'}, None)
    
    # Tune Stacking meta-learner
    base_estimators = [(name, clone(tuned_estimators[name])) for name in 
                       ['logistic_regression', 'random_forest', 'xgboost', 'lightgbm', 'catboost']
                       if name in tuned_estimators]
    if len(base_estimators) >= 2:
        stacking_model, stacking_params, stacking_score = tuner.tune_stacking(
            X_train, y_train, base_estimators, groups, timestamps, resampler, class_weight
        )
        results['stacking'] = (stacking_model, stacking_params, stacking_score)
    
    # Save all results
    tuner.save_all_results()
    tuner.save_best_params_csv()
    
    return results


# =============================================================================
# Export for Documentation Generation
# =============================================================================

def get_search_space_for_docs() -> Dict[str, Dict]:
    """Get search spaces formatted for documentation generation."""
    return SEARCH_SPACES.copy()


def get_tuning_protocol_summary() -> Dict[str, Any]:
    """Get tuning protocol summary for documentation."""
    return {
        'optimizer': 'Optuna-TPE',
        'sampler': 'TPESampler',
        'n_trials': N_TRIALS,
        'random_seed': RANDOM_SEED,
        'optimization_metric': 'MCC (Matthews Correlation Coefficient)',
        'early_stopping_rounds': EARLY_STOPPING_ROUNDS,
        'tuned_models': [
            'xgboost', 'lightgbm', 'catboost', 'random_forest',
            'logistic_regression', 'gradient_boosting', 'decision_tree', 
            'mlp', 'stacking (meta-learner only)'
        ],
        'not_tuned_models': ['naive_bayes', 'voting'],
        'inner_cv_rule': 'k = min(5, n_minority); if k < 3, use single 70/30 temporal split',
        'resampling': 'Applied only to training portion of each inner fold',
        'reference': 'Akiba et al. (2019). Optuna: A Next-generation Hyperparameter Optimization Framework. KDD 2019.'
    }
