#!/usr/bin/env python3
"""
Centralized Dataset Adequacy Thresholds

This module defines ALL adequacy thresholds in ONE place to ensure consistency
across the entire codebase and manuscript.

IMPORTANT: All thresholds are defined A PRIORI based ONLY on class counts,
NOT on model performance. This prevents any cherry-picking concerns.

=============================================================================
OFFICIAL THRESHOLD DEFINITIONS
=============================================================================

1. STATISTICAL ADEQUACY (for inclusion in primary analyses)
   ---------------------------------------------------------
   These thresholds determine whether a project-level combination is suitable
   for primary statistical comparisons (Friedman test, Nemenyi post-hoc, CD diagrams).

   PRIMARY:
       Train+Val buggy >= 20 AND Holdout buggy >= 10
       → Included in main statistical analyses

   EXPLORATORY:
       Train+Val buggy >= 5 AND Holdout buggy >= 3
       → Reported in Appendix, not in primary comparisons

   INSUFFICIENT:
       Below EXPLORATORY thresholds
       → Excluded from reporting due to unreliable metrics

2. TECHNICAL GUARDRAILS (prevent runtime errors)
   ----------------------------------------------
   These are minimum sample counts to prevent crashes, NOT for statistical purposes.

   MIN_TRAINVAL_SAMPLES = 50   → Ensures sufficient data for CV
   MIN_HOLDOUT_SAMPLES = 10    → Ensures holdout set is meaningful
   MIN_CV_FOLD_SAMPLES = 5     → Prevents empty folds
   MIN_CV_FOLD_MINORITY = 1    → Fallback for stratified splitting

=============================================================================
WHY TWO SETS OF THRESHOLDS?
=============================================================================

- STATISTICAL ADEQUACY thresholds (buggy-count based) determine analytical validity.
  A project can have 1000 samples but only 3 bugs → INSUFFICIENT for reliable F1/MCC.

- TECHNICAL GUARDRAILS (total sample count) prevent code failures.
  A project needs minimum samples for CV splits to work correctly.

Both must be satisfied, but they serve different purposes.

=============================================================================

Author: Bug Collector Team
Date: January 2026
"""

# =============================================================================
# STATISTICAL ADEQUACY THRESHOLDS (A PRIORI)
# Based on buggy (minority) class counts only
# =============================================================================

# PRIMARY: Suitable for main statistical analyses
MIN_TRAINVAL_MINORITY_PRIMARY = 20     # Train+Val buggy samples required
MIN_HOLDOUT_MINORITY_PRIMARY = 10      # Holdout buggy samples required

# EXPLORATORY: Reported but not in primary comparisons
MIN_TRAINVAL_MINORITY_EXPLORATORY = 5  # Below this, F1/MCC unreliable
MIN_HOLDOUT_MINORITY_EXPLORATORY = 3   # Below this, holdout metrics meaningless

# =============================================================================
# TECHNICAL GUARDRAILS
# Total sample counts to prevent runtime errors
# =============================================================================

MIN_TRAINVAL_SAMPLES = 50    # Minimum total Train+Val samples
MIN_HOLDOUT_SAMPLES = 10     # Minimum total Holdout samples
MIN_CV_FOLD_SAMPLES = 5      # Minimum samples per CV fold
MIN_CV_FOLD_MINORITY = 1     # Minimum minority per fold (fallback)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_statistical_status(trainval_buggy: int, holdout_buggy: int) -> str:
    """
    Determine statistical adequacy status based on buggy class counts.

    Args:
        trainval_buggy: Number of buggy samples in Train+Val set
        holdout_buggy: Number of buggy samples in Holdout set

    Returns:
        str: 'PRIMARY', 'EXPLORATORY', or 'INSUFFICIENT'
    """
    if (trainval_buggy >= MIN_TRAINVAL_MINORITY_PRIMARY and
        holdout_buggy >= MIN_HOLDOUT_MINORITY_PRIMARY):
        return 'PRIMARY'

    if (trainval_buggy >= MIN_TRAINVAL_MINORITY_EXPLORATORY and
        holdout_buggy >= MIN_HOLDOUT_MINORITY_EXPLORATORY):
        return 'EXPLORATORY'

    return 'INSUFFICIENT'


def passes_technical_guardrails(total_trainval: int, total_holdout: int) -> bool:
    """
    Check if sample counts pass technical guardrails.

    Args:
        total_trainval: Total samples in Train+Val set
        total_holdout: Total samples in Holdout set

    Returns:
        bool: True if guardrails are satisfied
    """
    return (total_trainval >= MIN_TRAINVAL_SAMPLES and
            total_holdout >= MIN_HOLDOUT_SAMPLES)


def get_threshold_documentation() -> str:
    """
    Return a formatted string documenting all thresholds.
    Useful for including in reports and manuscripts.
    """
    return f"""
Dataset Adequacy Thresholds (Defined A Priori)
==============================================

Statistical Adequacy (based on buggy class counts):
  - PRIMARY:     Train+Val buggy ≥ {MIN_TRAINVAL_MINORITY_PRIMARY}, Holdout buggy ≥ {MIN_HOLDOUT_MINORITY_PRIMARY}
  - EXPLORATORY: Train+Val buggy ≥ {MIN_TRAINVAL_MINORITY_EXPLORATORY}, Holdout buggy ≥ {MIN_HOLDOUT_MINORITY_EXPLORATORY}
  - INSUFFICIENT: Below EXPLORATORY thresholds

Technical Guardrails (total sample counts):
  - MIN_TRAINVAL_SAMPLES = {MIN_TRAINVAL_SAMPLES}
  - MIN_HOLDOUT_SAMPLES = {MIN_HOLDOUT_SAMPLES}
  - MIN_CV_FOLD_SAMPLES = {MIN_CV_FOLD_SAMPLES}
"""


if __name__ == '__main__':
    print(get_threshold_documentation())


