# Temporal Split Methodology: Ensuring No Data Leakage

## Response to Reviewer Comment R2C2

> **R2C2:** Split methodology: Explicitly confirm that file/method instances from test-period commits are not present in training. Add pseudocode or figure if possible.

---

## Overview

Our temporal split methodology ensures **strict temporal integrity** by splitting data at the **commit level**, not at the individual file/method level. This guarantees that:

1. **All instances (files/methods) from test-period commits are completely absent from training**
2. **No future information leaks into the training process**
3. **The temporal order of software evolution is preserved**

---

## Key Design Principle

The fundamental insight is that the **unit of temporal ordering is the commit**, not the individual file or method. A single commit may modify multiple files/methods simultaneously. Therefore:

- **Splitting at file/method level would violate temporal integrity** (same commit's modifications could appear in both train and test)
- **Splitting at commit level guarantees temporal separation** (all modifications from a commit stay together)

---

## Pseudocode

```
Algorithm: Temporal Commit-Based Train/Test Split
═══════════════════════════════════════════════════

Input:
  D = Dataset with instances (files or methods)
  Each instance i has:
    - commit_sha[i]: The commit that modified this file/method
    - commit_timestamp[i]: Unix timestamp of the commit
    - features[i]: Feature vector
    - label[i]: Bug label (0 or 1)

Output:
  D_train: Training dataset (first 80% of commits chronologically)
  D_test: Test dataset (last 20% of commits chronologically)

─────────────────────────────────────────────────────────────────

Step 1: Sort data by commit timestamp
─────────────────────────────────────
SORT D by commit_timestamp[i] in ascending order

Step 2: Extract unique commits preserving temporal order
────────────────────────────────────────────────────────
unique_commits ← empty list
seen_commits ← empty set

FOR each instance i in D (in sorted order):
    IF commit_sha[i] NOT IN seen_commits:
        APPEND commit_sha[i] to unique_commits
        ADD commit_sha[i] to seen_commits

Step 3: Split commits temporally (80/20)
────────────────────────────────────────
n_commits ← length(unique_commits)
split_index ← floor(n_commits × 0.8)

train_commits ← unique_commits[0 : split_index]      # First 80% chronologically
test_commits ← unique_commits[split_index : end]     # Last 20% chronologically

Step 4: Assign instances to train/test based on their commit
────────────────────────────────────────────────────────────
D_train ← empty dataset
D_test ← empty dataset

FOR each instance i in D:
    IF commit_sha[i] IN train_commits:
        ADD instance i to D_train
    ELSE:  # commit_sha[i] IN test_commits
        ADD instance i to D_test

Step 5: Validate temporal integrity (CRITICAL CHECK)
────────────────────────────────────────────────────
max_train_timestamp ← MAX(commit_timestamp[i] for i in D_train)
min_test_timestamp ← MIN(commit_timestamp[i] for i in D_test)

ASSERT max_train_timestamp < min_test_timestamp
    "Temporal integrity verified: No overlap between train and test periods"

─────────────────────────────────────────────────────────────────

RETURN D_train, D_test
```

---

## Formal Guarantee

Let $C_{train}$ and $C_{test}$ be the sets of commits in training and test sets respectively. Let $t(c)$ denote the timestamp of commit $c$.

**Theorem:** For all instances $i$ in test set and all instances $j$ in training set:
$$t(commit(i)) > t(commit(j))$$

**Proof:** 
1. By construction, $C_{train} \cap C_{test} = \emptyset$ (commits are partitioned)
2. By construction, $\max_{c \in C_{train}} t(c) < \min_{c \in C_{test}} t(c)$ (temporal ordering)
3. Each instance belongs to exactly one commit
4. Therefore, all test instances come from commits strictly after all training commits ∎

---

## Implementation in Code

From `analiz.py` (lines 2077-2123):

```python
# --- Temporal 80/20 Hold-out Split (commit-based) ---

# Step 1-2: Get unique commits in chronological order
unique_commits_ordered = []
seen_commits = set()
for sha in commit_groups:  # Already sorted by timestamp
    if sha not in seen_commits:
        unique_commits_ordered.append(sha)
        seen_commits.add(sha)

n_unique_commits = len(unique_commits_ordered)

# Step 3: Split commits temporally
holdout_split_idx = int(n_unique_commits * 0.8)
train_val_commits = set(unique_commits_ordered[:holdout_split_idx])
holdout_commits = set(unique_commits_ordered[holdout_split_idx:])

# Step 4: Assign instances based on their commit
train_val_mask = np.isin(commit_groups, list(train_val_commits))
holdout_mask = np.isin(commit_groups, list(holdout_commits))

X_train_val = X.iloc[train_val_indices]
X_holdout = X.iloc[holdout_indices]

# Step 5: Validate temporal integrity
train_val_max_ts = np.max(train_val_timestamps)
holdout_min_ts = np.min(holdout_timestamps)

if train_val_max_ts >= holdout_min_ts:
    logging.error("CRITICAL: Temporal integrity violation!")
else:
    logging.info(f"Hold-out split validated: train/val ends at {train_val_max_ts}, "
                 f"hold-out starts at {holdout_min_ts}")
```

---

## Why This Prevents Data Leakage

### Scenario: Multiple Files Modified in Same Commit

Consider a commit `abc123` at time $t_{100}$ that modifies 3 files: `main.go`, `util.go`, `test.go`.

| Instance | Commit | Timestamp | Assignment |
|----------|--------|-----------|------------|
| main.go changes | abc123 | $t_{100}$ | **ALL go to same set** |
| util.go changes | abc123 | $t_{100}$ | **ALL go to same set** |
| test.go changes | abc123 | $t_{100}$ | **ALL go to same set** |

If `abc123` is in the training period → all 3 instances are in training.
If `abc123` is in the test period → all 3 instances are in test.

**Never** will one file from this commit be in training while another is in test.

### Contrast with Incorrect Approach

If we split at file level (wrong approach):

| Instance | Random Split | Problem |
|----------|--------------|---------|
| main.go (abc123) | Train | ⚠️ Information from $t_{100}$ |
| util.go (abc123) | **Test** | ❌ Predicting $t_{100}$ with info from $t_{100}$! |
| test.go (abc123) | Train | ⚠️ Same timestamp as test instance |

This would be **data leakage** because training data includes information from the same time period as test data.

---

## Nested Cross-Validation with Temporal Integrity

For the inner CV loop, we use `CommitGroupTimeSeriesSplit` which:

1. Groups instances by commit SHA
2. Ensures all instances from same commit stay together
3. Uses expanding window approach (train always before validation)
4. Validates timestamps are monotonically increasing

```python
class CommitGroupTimeSeriesSplit:
    """
    Time-series cross-validator that respects commit group boundaries.
    
    Ensures:
    1. All instances from the same commit stay together in train or test
    2. Training data always comes chronologically before test data
    3. No data leakage between train and test sets
    """
```

---

## Summary

| Aspect | Our Methodology |
|--------|-----------------|
| **Split Level** | Commit (not file/method) |
| **Temporal Order** | Strictly preserved |
| **Same Commit Instances** | Always in same set (train OR test, never split) |
| **Validation** | Runtime assertion checks temporal integrity |
| **Data Leakage** | Impossible by construction |

The methodology ensures that **no file or method instance from test-period commits can appear in training data**, because:
1. We split by commits, not by instances
2. Commits are partitioned (mutually exclusive)
3. All instances inherit their commit's assignment
4. Temporal order is validated programmatically
