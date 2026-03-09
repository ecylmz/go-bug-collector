# Cross-Validation Strategies: Temporal vs Shuffle CV

## Overview

This document describes the two cross-validation strategies implemented in our study:
1. **Temporal CV**: Preserves chronological order of commits
2. **Shuffle CV**: Randomly shuffles data (baseline comparison)

Both strategies split data at the **commit level** to prevent data leakage.

---

## 1. Temporal CV (Default)

### Key Principle
- Training data always comes **chronologically before** validation/test data
- Commit groups are preserved (all instances from same commit stay together)
- Mimics real-world scenario: predict bugs in future code using past data

### Pseudocode

```
Algorithm: Temporal Cross-Validation with Commit Grouping
═════════════════════════════════════════════════════════

Input:
  D = Dataset sorted by commit_timestamp
  k = Number of CV folds
  
Output:
  Fold assignments [(train_indices, test_indices), ...]

─────────────────────────────────────────────────────────────────

Step 1: Extract unique commits in chronological order
─────────────────────────────────────────────────────
unique_commits ← []
seen ← {}
FOR each instance i in D:
    IF commit_sha[i] NOT IN seen:
        APPEND commit_sha[i] to unique_commits
        seen[commit_sha[i]] ← True

n_commits ← length(unique_commits)

Step 2: Generate k temporal folds (expanding window)
───────────────────────────────────────────────────
min_train_commits ← max(1, n_commits × 0.15)  # At least 15% for training
test_commits_per_fold ← (n_commits - min_train_commits) / k

FOR fold_idx FROM 0 TO k-1:
    
    # Calculate split point (expanding window)
    train_end_idx ← min_train_commits + (fold_idx × test_commits_per_fold)
    test_start_idx ← train_end_idx
    test_end_idx ← test_start_idx + test_commits_per_fold
    
    # Get commits for this fold
    train_commits ← unique_commits[0 : train_end_idx]
    test_commits ← unique_commits[test_start_idx : test_end_idx]
    
    # Map commits to instance indices
    train_indices ← [i for i in range(len(D)) if commit_sha[i] IN train_commits]
    test_indices ← [i for i in range(len(D)) if commit_sha[i] IN test_commits]
    
    # Validate temporal integrity
    ASSERT max(timestamp[train_indices]) < min(timestamp[test_indices])
    
    YIELD (train_indices, test_indices)

─────────────────────────────────────────────────────────────────

IMPORTANT: Within each fold:
  - train_indices preserves temporal order (NOT shuffled)
  - test_indices preserves temporal order (NOT shuffled)
  - Model sees data in order it was created
```

### Diagram: Expanding Window Temporal CV

```
Timeline: C1 → C2 → C3 → C4 → C5 → C6 → C7 → C8 → C9 → C10
          ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━→ Time

Fold 1:   [TRAIN: C1-C4    ][TEST: C5-C6 ]
Fold 2:   [TRAIN: C1-C6         ][TEST: C7-C8 ]
Fold 3:   [TRAIN: C1-C8              ][TEST: C9-C10]

Properties:
✓ Training always before testing
✓ Training set grows (expanding window)
✓ No overlap between train and test
✓ Same commit never split
```

---

## 2. Shuffle CV (Baseline Comparison)

### Key Principle
- **Holdout split is STILL TEMPORAL** (80/20 by commit timestamp)
- Only the **CV folds within the 80% train+val set** are shuffled
- Stratification preserves class balance in each fold
- Used as baseline to compare against temporal CV
- Shows effect of ignoring temporal order **within training data**

### Pseudocode

```
Algorithm: Stratified Shuffle Cross-Validation (with Temporal Holdout)
══════════════════════════════════════════════════════════════════════

Input:
  D = Full dataset sorted by commit_timestamp
  y = Labels (bug/not-bug)
  k = Number of CV folds
  random_state = Random seed for reproducibility
  
Output:
  CV fold assignments for Train+Val portion
  Holdout test set (temporal, unchanged)

─────────────────────────────────────────────────────────────────

Step 1: TEMPORAL Holdout Split (Same as Temporal CV!)
────────────────────────────────────────────────────
# Sort commits by timestamp
unique_commits ← sort_by_timestamp(get_unique_commits(D))
n_commits ← length(unique_commits)

# Split at 80% mark by TIME
holdout_split_idx ← floor(n_commits × 0.8)
train_val_commits ← unique_commits[0 : holdout_split_idx]      # First 80%
holdout_commits ← unique_commits[holdout_split_idx : n_commits] # Last 20%

# Get indices
train_val_indices ← [i for i in range(len(D)) if commit_sha[i] IN train_val_commits]
holdout_indices ← [i for i in range(len(D)) if commit_sha[i] IN holdout_commits]

# IMPORTANT: Holdout is ALWAYS temporal
ASSERT max(timestamp[train_val_indices]) < min(timestamp[holdout_indices])

Step 2: Shuffle CV ONLY on Train+Val (80%)
──────────────────────────────────────────
D_train_val ← D[train_val_indices]
y_train_val ← y[train_val_indices]

# Using sklearn's StratifiedKFold with shuffle
splitter ← StratifiedKFold(
    n_splits=k, 
    shuffle=True,           # <-- Shuffle within train+val only
    random_state=random_state
)

FOR (cv_train_idx, cv_val_idx) in splitter.split(D_train_val, y_train_val):
    
    # These indices are relative to D_train_val
    # Map back to original D indices if needed
    
    # Validate stratification (class ratio preserved)
    train_bug_ratio ← sum(y_train_val[cv_train_idx]) / len(cv_train_idx)
    val_bug_ratio ← sum(y_train_val[cv_val_idx]) / len(cv_val_idx)
    ASSERT abs(train_bug_ratio - val_bug_ratio) < tolerance
    
    YIELD (cv_train_idx, cv_val_idx)

─────────────────────────────────────────────────────────────────

KEY POINTS:
  ✓ Holdout test set (20%) is ALWAYS temporal (fair final evaluation)
  ✓ CV folds shuffle only within 80% train+val portion
  ✗ Within CV: temporal order ignored (validation may contain older commits)
  ✓ Final model trained on 80%, tested on temporal 20%
```

### Diagram: Shuffle CV (Corrected)

```
Full Timeline: C1 → C2 → C3 → C4 → C5 → C6 → C7 → C8 → C9 → C10
               ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━→ Time

Step 1: TEMPORAL Holdout Split (Same for Both CV Types!)
═══════════════════════════════════════════════════════

               [──────── Train+Val (80%) ────────][── Holdout (20%) ──]
               C1 → C2 → C3 → C4 → C5 → C6 → C7 → C8    C9 → C10
                                                   ↑
                                              TEMPORAL CUT

Step 2: Shuffle CV Folds (ONLY within Train+Val)
════════════════════════════════════════════════

Train+Val commits: C1, C2, C3, C4, C5, C6, C7, C8
After shuffle:     C4, C7, C1, C6, C3, C8, C2, C5  (random order)

Fold 1:   CV_TRAIN: C4,C7,C1,C6,C3,C8  |  CV_VAL: C2,C5
Fold 2:   CV_TRAIN: C4,C7,C1,C6,C2,C5  |  CV_VAL: C3,C8
...

Step 3: Final Evaluation (TEMPORAL!)
════════════════════════════════════

Train on: C1-C8 (all train+val)  →  Test on: C9-C10 (holdout, TEMPORAL)

Properties:
✓ Holdout test is ALWAYS temporal (future commits)
✓ Final evaluation is fair (no data leakage to holdout)
✗ CV validation may use "future" commits within train+val
✓ Balanced class distribution in each CV fold
```

---

## 3. Training Phase: Instance-Level Shuffling

### Important Clarification

Even in **Temporal CV**, once the train/test split is made:
- The **training set instances can be shuffled** during model fitting
- This is controlled by the ML model's internal settings
- Does NOT violate temporal integrity (split already made)

### Pseudocode: Model Training with Instance Shuffling

```
Algorithm: Model Training (Both CV Types)
═════════════════════════════════════════

Input:
  X_train = Training features (from temporal or shuffle CV split)
  y_train = Training labels
  model = ML model (e.g., RandomForest, LightGBM)
  
─────────────────────────────────────────────────────────────────

IMPORTANT DISTINCTION:

1. SPLIT-LEVEL (Temporal vs Shuffle):
   - Temporal CV: Commits assigned to train/test by TIME ORDER
   - Shuffle CV: Commits assigned to train/test RANDOMLY

2. TRAINING-LEVEL (Model Fitting):
   - Once X_train is determined, model may shuffle during SGD/batching
   - This is model-specific behavior (e.g., neural networks use shuffled batches)
   - Tree-based models typically don't shuffle (see full data at once)

─────────────────────────────────────────────────────────────────

# Model fitting (internal shuffling depends on model type)
model.fit(X_train, y_train)

# Example: Tree-based models (RandomForest, XGBoost, LightGBM)
# - Do NOT shuffle internally
# - See all training data at once
# - Bootstrap sampling is random, but that's different from shuffle

# Example: Neural Networks (MLP)
# - May shuffle data into mini-batches
# - shuffle parameter in DataLoader
# - Does not affect temporal integrity (split already made)

─────────────────────────────────────────────────────────────────

KEY POINT:
The temporal integrity is about WHAT data goes into training vs testing.
Once that split is made (at commit level), how the model internally
processes training data does not cause data leakage.
```

---

## 4. Complete Pipeline Comparison

### Temporal CV Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                      TEMPORAL CV PIPELINE                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  1. Data Preparation                                             │
│     ├── Load dataset D                                           │
│     ├── Sort by commit_timestamp (ASCENDING)                     │
│     └── Extract commit groups                                    │
│                                                                  │
│  2. Holdout Split (80/20 by commits)                            │
│     ├── First 80% commits → Train+Val set                       │
│     ├── Last 20% commits → Holdout Test set                     │
│     └── VALIDATE: max(train_ts) < min(test_ts)                  │
│                                                                  │
│  3. Nested CV on Train+Val (Temporal)                           │
│     ├── Outer loop: CommitGroupTimeSeriesSplit (k folds)        │
│     │   ├── Split commits temporally (expanding window)          │
│     │   ├── All instances from same commit stay together        │
│     │   └── VALIDATE temporal integrity each fold               │
│     │                                                            │
│     └── Inner loop: Time-aware hyperparameter tuning            │
│         ├── Further temporal split of outer-train               │
│         └── Select best hyperparameters                         │
│                                                                  │
│  4. Final Evaluation                                             │
│     ├── Train on full Train+Val with best hyperparams           │
│     └── Evaluate on Holdout Test (never seen during CV)         │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Shuffle CV Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                      SHUFFLE CV PIPELINE                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  1. Data Preparation                                             │
│     ├── Load dataset D                                           │
│     ├── Sort by commit_timestamp (for holdout split)            │
│     └── Extract labels for stratification                       │
│                                                                  │
│  2. Holdout Split (80/20 by commits, TEMPORAL - SAME!)          │
│     ├── First 80% commits → Train+Val set                       │
│     ├── Last 20% commits → Holdout Test set                     │
│     └── VALIDATE: max(train_val_ts) < min(holdout_ts)           │
│                                                                  │
│  3. Nested CV on Train+Val (Shuffle - ONLY HERE DIFFERENT)      │
│     ├── Outer loop: StratifiedKFold (k folds, shuffle=True)     │
│     │   ├── Random assignment within 80% train+val only         │
│     │   ├── Class balance preserved (stratified)                │
│     │   └── Temporal order ignored WITHIN train+val             │
│     │                                                            │
│     └── Inner loop: Standard hyperparameter tuning              │
│         ├── Random split of outer-train                         │
│         └── Select best hyperparameters                         │
│                                                                  │
│  4. Final Evaluation (TEMPORAL - SAME!)                         │
│     ├── Train on full Train+Val (80%) with best hyperparams     │
│     └── Evaluate on Holdout Test (20%, temporal, fair)          │
│                                                                  │
│  KEY DIFFERENCE: Only CV folds are shuffled, holdout is temporal│
└─────────────────────────────────────────────────────────────────┘
```

---

## 5. Implementation Reference

From `analiz.py`:

```python
# Check if shuffle CV is requested
use_shuffle_cv = cli_args.shuffle_cv if cli_args else False

if use_shuffle_cv:
    # Use stratified shuffle CV instead of temporal CV
    from sklearn.model_selection import StratifiedKFold
    outer_cv = StratifiedKFold(n_splits=outer_folds, shuffle=True, random_state=42)
    logging.info(f"Using Stratified Shuffle CV with {outer_folds} folds")
else:
    # Create temporal CV splitter for outer loop
    outer_cv = CommitGroupTimeSeriesSplit(
        n_splits=outer_folds,
        min_train_ratio=0.15,
        gap=0,
        min_class_ratio=min_class_ratio
    )
    logging.info(f"Using Temporal CV with {outer_folds} folds")
```

---

## 6. Summary Comparison

| Aspect | Temporal CV | Shuffle CV |
|--------|-------------|------------|
| **Holdout Split (20%)** | Temporal (last 20% commits) | **Temporal (same!)** |
| **CV Fold Order** | Chronological | Random |
| **CV Train/Val Relationship** | Train always before val | Mixed (random) |
| **Commit Grouping in CV** | Preserved | May split same commit |
| **Real-world Validity** | High (mimics deployment) | Medium (holdout still temporal) |
| **Class Balance in CV** | May vary by period | Stratified (balanced) |
| **Use Case** | Primary evaluation | Baseline comparison |
| **Flag** | Default | `--shuffle-cv` |
| **Final Test Fairness** | Fair (temporal holdout) | **Fair (temporal holdout)** |

---

## 7. Why Compare Both?

1. **Temporal CV** shows realistic CV performance (predicting chronologically future bugs)
2. **Shuffle CV** shows effect of ignoring temporal order within training
3. **Both use temporal holdout** - final test performance is comparable
4. **Gap in CV scores** indicates how much temporal ordering matters for model selection
5. If Shuffle CV >> Temporal CV: model selection benefits from temporal leakage
6. If Shuffle CV ≈ Temporal CV: features are temporally stable, order doesn't affect CV much

### Key Insight
The **holdout test scores** should be similar between both CV types because:
- Both use the same temporal 80/20 holdout split
- Final model is trained on all 80% and tested on temporal 20%
- The difference is only in **how hyperparameters are selected** during CV
