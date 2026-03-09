#!/bin/bash
# =============================================================================
# Curated Public Experiment Runner
# =============================================================================
# This script orchestrates reproducible experiment execution for the public
# `go-bug-collector` package.
#
# It can be used in two ways:
#   1. as a smoke-test runner for a smaller public reproduction pass
#   2. as a larger batch runner for the main experimental workflow
#
# Main responsibilities:
#   - run experiments across levels, projects, CV modes, resampling methods,
#     and feature configurations
#   - optionally run the Optuna-based optimization phase
#   - regenerate post-processing outputs such as adequacy tables,
#     statistical analyses, and summary figures
#
# Notes on scope:
#   - generated outputs are not versioned in Git
#   - some optional phases run only if their supporting scripts exist
#   - reviewer-only appendix tooling has been intentionally excluded from the
#     curated public repository
#
# Usage:
#   ./run_all_experiments.sh [OPTIONS]
#
# Common options:
#   --level LEVEL         Restrict execution to one level: commit, file, method
#   --project PROJECT     Restrict execution to one project
#   --quick               Run a smaller configuration for quick validation
#   --parallel N          Run up to N experiments in parallel
#   --dry-run             Print commands without executing them
#   --resume              Skip experiments with existing outputs
#   --optimize            Enable the Optuna-based optimization phase
#   --no-optimize         Explicitly disable the optimization phase
#   --no-hpo-robustness   Skip the optional HPO robustness phase
#   --hpo-trials N        Set the trial count for optional HPO robustness runs
#   --only-hpo-robustness Run only the optional HPO robustness phase
#   --help                Show the built-in usage summary
#
# Author: Bug Collector Team
# Date: January 2026
# =============================================================================

set -e  # Exit on error

# Activate virtual environment if it exists
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
    echo ">>> Virtual environment activated"
elif [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
    echo ">>> Virtual environment activated"
fi

# =============================================================================
# Configuration
# =============================================================================

# All available projects
ALL_PROJECTS=(
    "caddy" "compose" "consul" "fiber" "gin" "gitea" "grafana"
    "influxdb" "kubernetes" "minio" "nomad" "packer" "rclone"
    "terraform" "traefik" "vault"
)

# Levels
ALL_LEVELS=("commit" "file" "method")

# CV Types (temporal is default, shuffle requires flag)
CV_TYPES=("temporal" "shuffle")

# Resampling strategies (from analiz.py ALL_ACTUAL_RESAMPLING_METHODS)
RESAMPLING_STRATEGIES=("none" "smote" "random_under" "near_miss" "tomek" "random_over" "adasyn" "borderline" "smote_tomek" "smote_enn" "rose")

# Feature selection methods (empty string means no feature selection)
# Only using 'combine' (ensemble) and 'none' for efficiency
FEATURE_SELECTION=("" "combine")

# Feature sets: "full" (default) and "no_go_metrics" (excludes Go-specific features)
FEATURE_SETS=("full" "no_go_metrics")

# ML Methods - using 'all' runs all models at once
ML_METHODS="all"

# Parallel execution
PARALLEL_JOBS=1

# Flags
DRY_RUN=false
RESUME=false
QUICK_MODE=false
RUN_OPTIMIZE=false  # Skip full optimization phase (too many experiments) - use --optimize to enable
RUN_HPO_ROBUSTNESS=false  # Optional reviewer-oriented phase; disabled in the curated public repo by default
HPO_TRIALS=50  # Number of Optuna trials for HPO robustness check
SKIP_NORMAL_EXPERIMENTS=false  # Skip normal experiments (used with --only-hpo-robustness)

# Log directory
LOG_DIR="log/experiments"
RESULTS_LOG="$LOG_DIR/experiment_results.log"

# =============================================================================
# Argument Parsing
# =============================================================================

FILTER_LEVEL=""
FILTER_PROJECT=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --level)
            FILTER_LEVEL="$2"
            shift 2
            ;;
        --project)
            FILTER_PROJECT="$2"
            shift 2
            ;;
        --quick)
            QUICK_MODE=true
            shift
            ;;
        --parallel)
            PARALLEL_JOBS="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --resume)
            RESUME=true
            shift
            ;;
        --no-optimize)
            RUN_OPTIMIZE=false
            shift
            ;;
        --optimize)
            RUN_OPTIMIZE=true
            shift
            ;;
        --no-hpo-robustness)
            RUN_HPO_ROBUSTNESS=false
            shift
            ;;
        --hpo-trials)
            HPO_TRIALS="$2"
            shift 2
            ;;
        --only-hpo-robustness)
            # Skip normal experiments and optimization, only run HPO robustness check
            RUN_OPTIMIZE=false
            SKIP_NORMAL_EXPERIMENTS=true
            shift
            ;;
        --help)
            head -28 "$0" | tail -27
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# =============================================================================
# Quick Mode Configuration
# =============================================================================

if [ "$QUICK_MODE" = true ]; then
    echo ">>> Running in QUICK MODE (limited combinations)"
    ALL_PROJECTS=("influxdb" "kubernetes" "grafana")
    RESAMPLING_STRATEGIES=("none" "smote")
    FEATURE_SELECTION=("" "combine")
    FEATURE_SETS=("full")  # Only test full features in quick mode
fi

# =============================================================================
# Setup
# =============================================================================

mkdir -p "$LOG_DIR"

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Counters
TOTAL_EXPERIMENTS=0
COMPLETED_EXPERIMENTS=0
FAILED_EXPERIMENTS=0
SKIPPED_EXPERIMENTS=0

# =============================================================================
# Helper Functions
# =============================================================================

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

has_script() {
    [ -f "$1" ]
}

get_output_dir() {
    local level=$1
    local project=$2
    local cv_type=$3
    local resampling=$4
    local feature_sel=$5
    local feature_set=$6  # full or no_go_metrics

    local base_feature_set="${feature_set:-full}"
    if [ -n "$feature_sel" ]; then
        base_feature_set="fs_${feature_sel}"
    fi

    echo "results_${level}_level/${project}/${cv_type}/${base_feature_set}/${resampling}"
}

check_experiment_completed() {
    local output_dir=$1
    if [ -f "${output_dir}/analysis_summary.json" ]; then
        return 0  # Completed
    fi
    return 1  # Not completed
}

run_experiment() {
    local level=$1
    local project=$2
    local cv_type=$3
    local resampling=$4
    local feature_sel=$5
    local feature_set=$6  # full or no_go_metrics

    # Build command
    local cmd="python analiz.py --project $project --level $level --methods $ML_METHODS"

    # Add resampling
    if [ "$resampling" != "none" ]; then
        cmd="$cmd --resampling $resampling"
    else
        cmd="$cmd --resampling none"
    fi

    # Add CV type
    if [ "$cv_type" = "shuffle" ]; then
        cmd="$cmd --shuffle-cv"
    fi

    # Add feature selection
    if [ -n "$feature_sel" ]; then
        cmd="$cmd --select-feature $feature_sel"
    fi

    # Add exclude-go-metrics flag if feature_set is no_go_metrics
    if [ "$feature_set" = "no_go_metrics" ]; then
        cmd="$cmd --exclude-go-metrics"
    fi

    # Get output directory for checking
    local output_dir=$(get_output_dir "$level" "$project" "$cv_type" "$resampling" "$feature_sel" "$feature_set")

    # Resume mode - skip if completed
    if [ "$RESUME" = true ] && check_experiment_completed "$output_dir"; then
        log_warning "Skipping (already completed): $output_dir"
        ((SKIPPED_EXPERIMENTS++))
        return 0
    fi

    # Log file for this experiment
    local log_file="$LOG_DIR/${level}_${project}_${cv_type}_${feature_set}_${resampling}_${feature_sel:-none}.log"

    echo "----------------------------------------"
    log_info "Running: $cmd"
    log_info "Output: $output_dir"
    log_info "Log: $log_file"

    if [ "$DRY_RUN" = true ]; then
        echo "$cmd"
        return 0
    fi

    # Run experiment
    local start_time=$(date +%s)

    if $cmd > "$log_file" 2>&1; then
        local end_time=$(date +%s)
        local duration=$((end_time - start_time))
        log_success "Completed in ${duration}s: $project/$level/$cv_type/$feature_set/$resampling/${feature_sel:-none}"
        ((COMPLETED_EXPERIMENTS++))
        echo "SUCCESS,$project,$level,$cv_type,$feature_set,$resampling,${feature_sel:-none},$duration" >> "$RESULTS_LOG"
    else
        local end_time=$(date +%s)
        local duration=$((end_time - start_time))
        log_error "Failed after ${duration}s: $project/$level/$cv_type/$feature_set/$resampling/${feature_sel:-none}"
        log_error "Check log: $log_file"
        ((FAILED_EXPERIMENTS++))
        echo "FAILED,$project,$level,$cv_type,$feature_set,$resampling,${feature_sel:-none},$duration" >> "$RESULTS_LOG"
    fi
}

# Parallel execution helper
run_experiment_parallel() {
    # Same as run_experiment but for parallel execution
    # Writes to individual status files instead of shared counters
    local level=$1
    local project=$2
    local cv_type=$3
    local resampling=$4
    local feature_sel=$5
    local feature_set=$6  # full or no_go_metrics

    local cmd="python analiz.py --project $project --level $level --methods $ML_METHODS"

    if [ "$resampling" != "none" ]; then
        cmd="$cmd --resampling $resampling"
    else
        cmd="$cmd --resampling none"
    fi

    if [ "$cv_type" = "shuffle" ]; then
        cmd="$cmd --shuffle-cv"
    fi

    if [ -n "$feature_sel" ]; then
        cmd="$cmd --select-feature $feature_sel"
    fi

    # Add exclude-go-metrics flag if feature_set is no_go_metrics
    if [ "$feature_set" = "no_go_metrics" ]; then
        cmd="$cmd --exclude-go-metrics"
    fi

    local output_dir=$(get_output_dir "$level" "$project" "$cv_type" "$resampling" "$feature_sel" "$feature_set")
    local log_file="$LOG_DIR/${level}_${project}_${cv_type}_${feature_set}_${resampling}_${feature_sel:-none}.log"
    local status_file="$LOG_DIR/.status_${level}_${project}_${cv_type}_${feature_set}_${resampling}_${feature_sel:-none}"

    # Resume mode
    if [ "$RESUME" = true ] && check_experiment_completed "$output_dir"; then
        echo "SKIPPED" > "$status_file"
        return 0
    fi

    local start_time=$(date +%s)

    if $cmd > "$log_file" 2>&1; then
        local end_time=$(date +%s)
        local duration=$((end_time - start_time))
        echo "SUCCESS,$project,$level,$cv_type,$feature_set,$resampling,${feature_sel:-none},$duration" >> "$RESULTS_LOG"
        echo "SUCCESS" > "$status_file"
    else
        local end_time=$(date +%s)
        local duration=$((end_time - start_time))
        echo "FAILED,$project,$level,$cv_type,$feature_set,$resampling,${feature_sel:-none},$duration" >> "$RESULTS_LOG"
        echo "FAILED" > "$status_file"
    fi
}

# Export functions for parallel execution
export -f run_experiment_parallel get_output_dir check_experiment_completed
export LOG_DIR RESULTS_LOG ML_METHODS RESUME

# =============================================================================
# Main Execution
# =============================================================================

echo "============================================================================="
echo "                    Bug Prediction Experiment Runner"
echo "============================================================================="
echo ""

# Initialize results log
echo "STATUS,PROJECT,LEVEL,CV_TYPE,FEATURE_SET,RESAMPLING,FEATURE_SEL,DURATION_SEC" > "$RESULTS_LOG"

# Calculate total experiments
for level in "${ALL_LEVELS[@]}"; do
    [ -n "$FILTER_LEVEL" ] && [ "$level" != "$FILTER_LEVEL" ] && continue

    for project in "${ALL_PROJECTS[@]}"; do
        [ -n "$FILTER_PROJECT" ] && [ "$project" != "$FILTER_PROJECT" ] && continue

        for cv_type in "${CV_TYPES[@]}"; do
            for feature_set in "${FEATURE_SETS[@]}"; do
                for resampling in "${RESAMPLING_STRATEGIES[@]}"; do
                    for feature_sel in "${FEATURE_SELECTION[@]}"; do
                        ((TOTAL_EXPERIMENTS++))
                    done
                done
            done
        done
    done
done

echo "Configuration:"
echo "  - Levels: ${ALL_LEVELS[*]}"
echo "  - Projects: ${#ALL_PROJECTS[@]} projects"
echo "  - CV Types: ${CV_TYPES[*]}"
echo "  - Feature Sets: ${FEATURE_SETS[*]}"
echo "  - Resampling: ${RESAMPLING_STRATEGIES[*]}"
echo "  - Feature Selection: ${FEATURE_SELECTION[*]:-none} (${#FEATURE_SELECTION[@]} options)"
echo "  - Total Experiments: $TOTAL_EXPERIMENTS"
echo "  - Parallel Jobs: $PARALLEL_JOBS"
echo "  - Dry Run: $DRY_RUN"
echo "  - Resume Mode: $RESUME"
echo "  - Optimization Phase: $RUN_OPTIMIZE"
echo "  - HPO Robustness Check: $RUN_HPO_ROBUSTNESS (trials: $HPO_TRIALS)"
echo "  - Skip Normal Experiments: $SKIP_NORMAL_EXPERIMENTS"
echo ""

if [ "$DRY_RUN" = false ] && [ "$SKIP_NORMAL_EXPERIMENTS" = false ]; then
    read -p "Press Enter to start experiments (Ctrl+C to cancel)..."
fi

START_TIME=$(date +%s)

# Run experiments
CURRENT=0

# Skip normal experiments if --only-hpo-robustness is set
if [ "$SKIP_NORMAL_EXPERIMENTS" = true ]; then
    echo ""
    log_info "Skipping normal experiments (--only-hpo-robustness mode)"
    echo ""
elif [ "$PARALLEL_JOBS" -gt 1 ] && [ "$DRY_RUN" = false ]; then
    # =============================================================================
    # Parallel Execution Mode
    # =============================================================================
    echo ""
    log_info "Running in PARALLEL mode with $PARALLEL_JOBS jobs"
    echo ""

    # Create a file with all experiment combinations
    JOBS_FILE="$LOG_DIR/.parallel_jobs.txt"
    > "$JOBS_FILE"

    for level in "${ALL_LEVELS[@]}"; do
        [ -n "$FILTER_LEVEL" ] && [ "$level" != "$FILTER_LEVEL" ] && continue

        for project in "${ALL_PROJECTS[@]}"; do
            [ -n "$FILTER_PROJECT" ] && [ "$project" != "$FILTER_PROJECT" ] && continue

            for cv_type in "${CV_TYPES[@]}"; do
                for feature_set in "${FEATURE_SETS[@]}"; do
                    for resampling in "${RESAMPLING_STRATEGIES[@]}"; do
                        for feature_sel in "${FEATURE_SELECTION[@]}"; do
                            echo "$level $project $cv_type $resampling $feature_sel $feature_set" >> "$JOBS_FILE"
                        done
                    done
                done
            done
        done
    done

    # Check if GNU parallel is available
    if command -v parallel &> /dev/null; then
        log_info "Using GNU parallel"

        # Run with GNU parallel
        cat "$JOBS_FILE" | parallel --jobs "$PARALLEL_JOBS" --colsep ' ' \
            'run_experiment_parallel {1} {2} {3} {4} {5} {6}'

    else
        log_info "GNU parallel not found, using background jobs with job control"

        # Simple parallel execution with background jobs
        RUNNING_JOBS=0

        while IFS=' ' read -r level project cv_type resampling feature_sel feature_set; do
            ((CURRENT++))
            echo ">>> Starting experiment $CURRENT/$TOTAL_EXPERIMENTS: $project/$level/$cv_type/$feature_set/$resampling/${feature_sel:-none}"

            # Run in background
            run_experiment_parallel "$level" "$project" "$cv_type" "$resampling" "$feature_sel" "$feature_set" &
            ((RUNNING_JOBS++))

            # Wait if we have reached max parallel jobs
            if [ "$RUNNING_JOBS" -ge "$PARALLEL_JOBS" ]; then
                wait -n 2>/dev/null || wait
                ((RUNNING_JOBS--))
            fi
        done < "$JOBS_FILE"

        # Wait for remaining jobs
        log_info "Waiting for remaining jobs to complete..."
        wait
    fi

    # Count results from status files
    COMPLETED_EXPERIMENTS=$(find "$LOG_DIR" -name ".status_*" -exec cat {} \; | grep -c "SUCCESS" || echo 0)
    FAILED_EXPERIMENTS=$(find "$LOG_DIR" -name ".status_*" -exec cat {} \; | grep -c "FAILED" || echo 0)
    SKIPPED_EXPERIMENTS=$(find "$LOG_DIR" -name ".status_*" -exec cat {} \; | grep -c "SKIPPED" || echo 0)

    # Cleanup status files
    rm -f "$LOG_DIR"/.status_* "$JOBS_FILE"

else
    # =============================================================================
    # Sequential Execution Mode
    # =============================================================================
    for level in "${ALL_LEVELS[@]}"; do
        [ -n "$FILTER_LEVEL" ] && [ "$level" != "$FILTER_LEVEL" ] && continue

        echo ""
        echo "============================================================================="
        echo "                         Level: $level"
        echo "============================================================================="

        for project in "${ALL_PROJECTS[@]}"; do
            [ -n "$FILTER_PROJECT" ] && [ "$project" != "$FILTER_PROJECT" ] && continue

            echo ""
            log_info "Project: $project"

            for cv_type in "${CV_TYPES[@]}"; do
                for feature_set in "${FEATURE_SETS[@]}"; do
                    for resampling in "${RESAMPLING_STRATEGIES[@]}"; do
                        for feature_sel in "${FEATURE_SELECTION[@]}"; do
                            ((CURRENT++))
                            echo ""
                            echo ">>> Experiment $CURRENT/$TOTAL_EXPERIMENTS"

                            run_experiment "$level" "$project" "$cv_type" "$resampling" "$feature_sel" "$feature_set"
                        done
                    done
                done
            done
        done
    done
fi

END_TIME=$(date +%s)
TOTAL_DURATION=$((END_TIME - START_TIME))

# =============================================================================
# Summary
# =============================================================================

echo ""
echo "============================================================================="
echo "                           Experiment Summary"
echo "============================================================================="
echo ""
echo "Total Experiments: $TOTAL_EXPERIMENTS"
echo "Completed: $COMPLETED_EXPERIMENTS"
echo "Failed: $FAILED_EXPERIMENTS"
echo "Skipped: $SKIPPED_EXPERIMENTS"
echo "Total Duration: ${TOTAL_DURATION}s ($(($TOTAL_DURATION / 60)) minutes)"
echo ""
echo "Results log: $RESULTS_LOG"
echo ""

# =============================================================================
# Generate Reports, Statistical Analysis, and Academic Outputs
# =============================================================================

if [ "$DRY_RUN" = false ] && [ $COMPLETED_EXPERIMENTS -gt 0 ] && [ "$SKIP_NORMAL_EXPERIMENTS" = false ]; then
    echo "============================================================================="
    echo "                     Generating Reports and Analysis"
    echo "============================================================================="
    echo ""

    # 1. Generate comprehensive reports (results.md and plots for each project)
    log_info "Step 1/9: Generating comprehensive reports..."
    for level in "${ALL_LEVELS[@]}"; do
        [ -n "$FILTER_LEVEL" ] && [ "$level" != "$FILTER_LEVEL" ] && continue

        log_info "  Generating reports for $level level..."
        python analiz.py --level $level --generate-reports 2>/dev/null || log_warning "Failed to generate reports for $level"
    done

    # 2. Generate statistical analysis (Friedman test, CD diagrams, effect sizes)
    log_info "Step 2/9: Generating statistical analysis..."
    for level in "${ALL_LEVELS[@]}"; do
        [ -n "$FILTER_LEVEL" ] && [ "$level" != "$FILTER_LEVEL" ] && continue

        for cv_type in "${CV_TYPES[@]}"; do
            for feature_set in "${FEATURE_SETS[@]}"; do
                log_info "  Statistical analysis: $level / $cv_type / $feature_set"
                python statistical_analysis.py --level $level --cv-type $cv_type --feature-set $feature_set 2>/dev/null || \
                    log_warning "Failed statistical analysis for $level/$cv_type/$feature_set"
            done
        done
    done

    # 3. Generate dataset adequacy table (for statistical analysis inclusion criteria)
    log_info "Step 3/9: Generating dataset adequacy table..."
    python generate_adequacy_table.py 2>/dev/null || log_warning "Failed to generate adequacy table"

    # 4. Generate project inclusion matrix (PRIMARY/EXPLORATORY/INSUFFICIENT status)
    log_info "Step 4/9: Generating project inclusion matrix..."
    python generate_inclusion_matrix.py 2>/dev/null || log_warning "Failed to generate inclusion matrix"

    # 5. Generate exploratory results table (Appendix - non-PRIMARY projects)
    log_info "Step 5/9: Generating exploratory results table..."
    python generate_exploratory_table.py 2>/dev/null || log_warning "Failed to generate exploratory table"

    # 6. Generate resampling reproducibility table (synthetic sample counts)
    log_info "Step 6/9: Generating resampling reproducibility table..."
    python generate_resampling_table.py 2>/dev/null || log_warning "Failed to generate resampling table"

    # 7. Generate academic tables and figures
    log_info "Step 7/9: Generating academic tables and figures..."
    python academic_figures.py 2>/dev/null || log_warning "Failed to generate academic figures"

    # 8. Generate Top 5 Most Influential Features tables per level
    log_info "Step 8/9: Generating feature importance tables..."
    python generate_feature_importance_table.py 2>/dev/null || log_warning "Failed to generate feature importance tables"

    # 9. Generate summary plots (legacy)
    log_info "Step 9/9: Generating summary plots..."
    python analiz.py --generate-summary-plots 2>/dev/null || log_warning "Failed to generate summary plots"

    echo ""
    log_success "All reports, analysis, and figures generated!"
    echo ""
    echo "Generated outputs:"
    echo "  - Project reports:       results_{level}_level/{project}/{cv_type}/{feature_set}/results.md"
    echo "  - Statistical analysis:  academic_outputs/statistical_analysis/{level}/{cv_type}/{feature_set}/"
    echo "  - Dataset adequacy:      academic_outputs/table_dataset_adequacy.{md,tex,csv}"
    echo "  - Inclusion matrix:      academic_outputs/table_project_inclusion_matrix.{md,tex,csv}"
    echo "  - Exploratory results:   academic_outputs/table_exploratory_results.{md,tex,csv}"
    echo "  - Resampling table:      academic_outputs/resampling_reproducibility.{md,tex,csv}"
    echo "  - Feature importance:    academic_outputs/table_top_features_{commit,file,method}.md"
    echo "  - Academic outputs:      academic_outputs/"
fi

# =============================================================================
# Phase 2: Optimization Experiments (Optuna-TPE Hyperparameter Tuning)
# =============================================================================

if [ "$DRY_RUN" = false ] && [ "$RUN_OPTIMIZE" = true ] && [ "$SKIP_NORMAL_EXPERIMENTS" = false ]; then
    echo ""
    echo "============================================================================="
    echo "         Phase 2: Optimization Experiments (Optuna-TPE Tuning)"
    echo "============================================================================="
    echo ""
    echo "Starting hyperparameter optimization using Optuna-TPE with 100 trials..."
    echo "This phase will tune hyperparameters for each model using MCC as objective."
    echo "NOTE: Only PRIMARY projects (meeting statistical adequacy thresholds) are included."
    echo ""

    # Reset counters for optimization phase
    OPTIM_TOTAL=0
    OPTIM_COMPLETED=0
    OPTIM_FAILED=0
    OPTIM_SKIPPED=0

    # Create optimization log
    OPTIM_RESULTS_LOG="$LOG_DIR/optimization_results.log"
    echo "STATUS,PROJECT,LEVEL,CV_TYPE,FEATURE_SET,RESAMPLING,FEATURE_SEL,DURATION_SEC" > "$OPTIM_RESULTS_LOG"

    # Function to get PRIMARY projects for a level
    get_primary_projects() {
        local level=$1
        python -c "from adequacy_filter import get_primary_projects; print(' '.join(get_primary_projects('$level')))" 2>/dev/null
    }

    # Calculate total optimization experiments (only PRIMARY projects)
    for level in "${ALL_LEVELS[@]}"; do
        [ -n "$FILTER_LEVEL" ] && [ "$level" != "$FILTER_LEVEL" ] && continue

        # Get PRIMARY projects for this level
        PRIMARY_PROJECTS=$(get_primary_projects "$level")

        for project in $PRIMARY_PROJECTS; do
            [ -n "$FILTER_PROJECT" ] && [ "$project" != "$FILTER_PROJECT" ] && continue

            for cv_type in "${CV_TYPES[@]}"; do
                for feature_set in "${FEATURE_SETS[@]}"; do
                    for resampling in "${RESAMPLING_STRATEGIES[@]}"; do
                        for feature_sel in "${FEATURE_SELECTION[@]}"; do
                            ((OPTIM_TOTAL++))
                        done
                    done
                done
            done
        done
    done

    echo "Optimization experiments to run: $OPTIM_TOTAL"
    echo ""

    OPTIM_START_TIME=$(date +%s)
    OPTIM_CURRENT=0

    # Run optimization experiments (only PRIMARY projects)
    for level in "${ALL_LEVELS[@]}"; do
        [ -n "$FILTER_LEVEL" ] && [ "$level" != "$FILTER_LEVEL" ] && continue

        echo ""
        echo "============================================================================="
        echo "                    Optimization - Level: $level"
        echo "============================================================================="

        # Get PRIMARY projects for this level
        PRIMARY_PROJECTS=$(get_primary_projects "$level")
        echo "PRIMARY projects for $level: $PRIMARY_PROJECTS"

        for project in $PRIMARY_PROJECTS; do
            [ -n "$FILTER_PROJECT" ] && [ "$project" != "$FILTER_PROJECT" ] && continue

            echo ""
            log_info "Optimizing Project: $project"

            for cv_type in "${CV_TYPES[@]}"; do
                for feature_set in "${FEATURE_SETS[@]}"; do
                    for resampling in "${RESAMPLING_STRATEGIES[@]}"; do
                        for feature_sel in "${FEATURE_SELECTION[@]}"; do
                            ((OPTIM_CURRENT++))
                            echo ""
                            echo ">>> Optimization $OPTIM_CURRENT/$OPTIM_TOTAL"

                            # Build optimization command
                            local_cmd="python analiz.py --project $project --level $level --methods $ML_METHODS --optimize"

                            if [ "$resampling" != "none" ]; then
                                local_cmd="$local_cmd --resampling $resampling"
                            else
                                local_cmd="$local_cmd --resampling none"
                            fi

                            if [ "$cv_type" = "shuffle" ]; then
                                local_cmd="$local_cmd --shuffle-cv"
                            fi

                            if [ -n "$feature_sel" ]; then
                                local_cmd="$local_cmd --select-feature $feature_sel"
                            fi

                            if [ "$feature_set" = "no_go_metrics" ]; then
                                local_cmd="$local_cmd --exclude-go-metrics"
                            fi

                            # Log file for this optimization experiment
                            local_log_file="$LOG_DIR/optim_${level}_${project}_${cv_type}_${feature_set}_${resampling}_${feature_sel:-none}.log"

                            echo "----------------------------------------"
                            log_info "Running: $local_cmd"
                            log_info "Log: $local_log_file"

                            # Run optimization experiment
                            local_start_time=$(date +%s)

                            if $local_cmd > "$local_log_file" 2>&1; then
                                local_end_time=$(date +%s)
                                local_duration=$((local_end_time - local_start_time))
                                log_success "Completed in ${local_duration}s: $project/$level/$cv_type/$feature_set/$resampling/${feature_sel:-none}"
                                ((OPTIM_COMPLETED++))
                                echo "SUCCESS,$project,$level,$cv_type,$feature_set,$resampling,${feature_sel:-none},$local_duration" >> "$OPTIM_RESULTS_LOG"
                            else
                                local_end_time=$(date +%s)
                                local_duration=$((local_end_time - local_start_time))
                                log_error "Failed after ${local_duration}s: $project/$level/$cv_type/$feature_set/$resampling/${feature_sel:-none}"
                                log_error "Check log: $local_log_file"
                                ((OPTIM_FAILED++))
                                echo "FAILED,$project,$level,$cv_type,$feature_set,$resampling,${feature_sel:-none},$local_duration" >> "$OPTIM_RESULTS_LOG"
                            fi
                        done
                    done
                done
            done
        done
    done

    OPTIM_END_TIME=$(date +%s)
    OPTIM_DURATION=$((OPTIM_END_TIME - OPTIM_START_TIME))

    echo ""
    echo "============================================================================="
    echo "                     Optimization Phase Summary"
    echo "============================================================================="
    echo ""
    echo "Total Optimization Experiments: $OPTIM_TOTAL"
    echo "Completed: $OPTIM_COMPLETED"
    echo "Failed: $OPTIM_FAILED"
    echo "Duration: ${OPTIM_DURATION}s ($(($OPTIM_DURATION / 60)) minutes)"
    echo ""
    echo "Optimization results log: $OPTIM_RESULTS_LOG"
    echo "Tuning results saved to: results_{level}_level/{project}/{cv_type}/{feature_set}/{resampling}/optimization_results/"
fi

# =============================================================================
# Phase 3: HPO Robustness Check (Representative Scenarios)
# =============================================================================

if [ "$DRY_RUN" = false ] && [ "$RUN_HPO_ROBUSTNESS" = true ] && has_script "run_hpo_robustness_check.py" && has_script "generate_hpo_academic_outputs.py"; then
    echo ""
    echo "============================================================================="
    echo "     Phase 3: HPO Robustness Check (Representative Scenarios)"
    echo "============================================================================="
    echo ""
    echo "Running hyperparameter tuning robustness check on representative scenarios."
    echo "This runs default vs tuned comparison for boosting models (XGBoost, LightGBM, CatBoost)"
    echo "with pre-defined resampling conditions (None, SMOTE) on:"
    echo "  - Large-scale scenario (max samples) per level"
    echo "  - Worst-case imbalance scenario (min minority ratio) per level"
    echo ""

    HPO_START_TIME=$(date +%s)

    # Build levels argument
    HPO_LEVELS="commit,file,method"
    if [ -n "$FILTER_LEVEL" ]; then
        HPO_LEVELS="$FILTER_LEVEL"
    fi

    # Run HPO robustness check
    HPO_CMD="python run_hpo_robustness_check.py --data_root . --output_dir hpo_robustness_results --levels $HPO_LEVELS --trials $HPO_TRIALS --seed 42"

    echo "Running: $HPO_CMD"
    echo ""

    HPO_LOG_FILE="$LOG_DIR/hpo_robustness_check.log"

    if $HPO_CMD > "$HPO_LOG_FILE" 2>&1; then
        HPO_END_TIME=$(date +%s)
        HPO_DURATION=$((HPO_END_TIME - HPO_START_TIME))
        log_success "HPO Robustness Check completed in ${HPO_DURATION}s"
        echo ""
        echo "HPO Robustness Check outputs:"
        echo "  - Selected scenarios: hpo_robustness_results/selected_scenarios.csv"
        echo "  - Results: hpo_robustness_results/robustness_check_results.csv"
        echo "  - Best params: hpo_robustness_results/best_params.json"
        echo "  - LaTeX table: hpo_robustness_results/robustness_check_table.tex"
        echo "  - Log: $HPO_LOG_FILE"
    else
        log_error "HPO Robustness Check failed. Check log: $HPO_LOG_FILE"
    fi

    # Generate HPO academic outputs (tables, figures, statistical tests)
    echo ""
    log_info "Generating HPO academic outputs (tables, figures, Wilcoxon tests)..."
    HPO_ACADEMIC_CMD="python generate_hpo_academic_outputs.py"
    HPO_ACADEMIC_LOG="$LOG_DIR/hpo_academic_outputs.log"

    if $HPO_ACADEMIC_CMD > "$HPO_ACADEMIC_LOG" 2>&1; then
        log_success "HPO academic outputs generated successfully"
        echo "  - HPO tables: academic_outputs/hpo/tables/"
        echo "  - HPO figures: academic_outputs/hpo/figures/"
        echo "  - Wilcoxon tests: academic_outputs/hpo/data/wilcoxon_test_results.json"
    else
        log_warning "HPO academic outputs generation failed. Check log: $HPO_ACADEMIC_LOG"
    fi

    # Also copy main results file to academic_outputs root
    if [ -f "hpo_robustness_results/robustness_check_results.csv" ]; then
        cp hpo_robustness_results/robustness_check_results.csv academic_outputs/hpo_robustness_results.csv
        log_info "Copied HPO results to academic_outputs/hpo_robustness_results.csv"
    fi
fi

if [ "$DRY_RUN" = false ] && [ "$RUN_HPO_ROBUSTNESS" = true ] && ! has_script "run_hpo_robustness_check.py"; then
    log_info "Skipping HPO robustness phase because optional reviewer-only scripts are not included in this curated repo."
fi

# =============================================================================
# Phase 4: Generate Bootstrap CI Figures (Academic Publication)
# =============================================================================

if [ "$DRY_RUN" = false ] && has_script "generate_bootstrap_ci_figures.py"; then
    echo ""
    echo "============================================================================="
    echo "     Phase 4: Generating Bootstrap CI Figures (Academic Publication)"
    echo "============================================================================="
    echo ""

    # Only generate figures if results exist
    if [ -d "results_commit_level" ] || [ -d "results_file_level" ] || [ -d "results_method_level" ]; then
        BOOTSTRAP_CMD="python generate_bootstrap_ci_figures.py --results_dir . --output_dir academic_outputs/bootstrap_ci"

        # Add level filter if specified
        if [ -n "$FILTER_LEVEL" ]; then
            BOOTSTRAP_CMD="$BOOTSTRAP_CMD --levels $FILTER_LEVEL"
        fi

        echo "Running: $BOOTSTRAP_CMD"
        echo ""

        BOOTSTRAP_LOG_FILE="$LOG_DIR/bootstrap_ci_figures.log"

        if $BOOTSTRAP_CMD > "$BOOTSTRAP_LOG_FILE" 2>&1; then
            log_success "Bootstrap CI figures generated successfully"
            echo ""
            echo "Bootstrap CI Figure outputs:"
            echo "  - Forest plots: academic_outputs/bootstrap_ci/forest_plot_*.pdf"
            echo "  - CI heatmaps: academic_outputs/bootstrap_ci/ci_heatmap_*.pdf"
            echo "  - Model comparison: academic_outputs/bootstrap_ci/model_comparison_ci_*.pdf"
            echo "  - Distribution plots: academic_outputs/bootstrap_ci/distribution_ci_*.pdf"
            echo "  - LaTeX tables: academic_outputs/bootstrap_ci/ci_table_*.tex"
            echo "  - Log: $BOOTSTRAP_LOG_FILE"
        else
            log_warning "Bootstrap CI figure generation failed or no data available. Check log: $BOOTSTRAP_LOG_FILE"
        fi
    else
        log_info "No results directories found. Skipping bootstrap CI figure generation."
        log_info "Run experiments first with Phase 1 to generate results."
    fi
fi

if [ "$DRY_RUN" = false ] && ! has_script "generate_bootstrap_ci_figures.py"; then
    log_info "Skipping bootstrap CI figure generation (optional script not included in curated repo)."
fi

# =============================================================================
# Phase 5: MCC=0 Diagnostic Audit
# =============================================================================

if [ "$DRY_RUN" = false ] && has_script "diagnose_mcc_zero.py"; then
    echo ""
    echo "============================================================================="
    echo "     Phase 5: MCC=0 Diagnostic Audit"
    echo "============================================================================="
    echo ""

    # Run MCC=0 diagnosis
    log_info "Running MCC=0 diagnostic analysis..."
    DIAG_CMD="python diagnose_mcc_zero.py"
    DIAG_LOG_FILE="$LOG_DIR/diagnose_mcc_zero.log"

    if $DIAG_CMD > "$DIAG_LOG_FILE" 2>&1; then
        log_success "MCC=0 diagnostic analysis completed"
        echo "  - MCC zero cases: academic_outputs/diagnostics/mcc_zero_cases.csv"
        echo "  - Metrics report: academic_outputs/metrics_definition_report.md"
    else
        log_warning "MCC=0 diagnostic analysis failed. Check log: $DIAG_LOG_FILE"
    fi

    # Run detailed MCC=0 audit with threshold sensitivity
    log_info "Running detailed MCC=0 audit with threshold sensitivity..."
    DETAILED_AUDIT_CMD="python audit_mcc_zero_detailed.py"
    DETAILED_AUDIT_LOG="$LOG_DIR/audit_mcc_zero_detailed.log"

    if $DETAILED_AUDIT_CMD > "$DETAILED_AUDIT_LOG" 2>&1; then
        log_success "Detailed MCC=0 audit completed"
        echo "  - Root causes: academic_outputs/diagnostics/mcc0_root_causes.csv"
        echo "  - Rate analysis: academic_outputs/diagnostics/mcc0_rate_analysis.csv"
        echo "  - Threshold sensitivity: academic_outputs/diagnostics/mcc0_threshold_sensitivity.csv"
        echo "  - Detailed report: academic_outputs/diagnostics/mcc0_detailed_audit.md"
    else
        log_warning "Detailed MCC=0 audit failed. Check log: $DETAILED_AUDIT_LOG"
    fi

    # Run comprehensive MCC=0 audit (leakage check + reviewer summary)
    log_info "Running comprehensive MCC=0 audit (with leakage check)..."
    COMPREHENSIVE_CMD="python audit_mcc_zero_comprehensive.py"
    COMPREHENSIVE_LOG="$LOG_DIR/audit_mcc_zero_comprehensive.log"

    if $COMPREHENSIVE_CMD > "$COMPREHENSIVE_LOG" 2>&1; then
        log_success "Comprehensive MCC=0 audit completed"
        echo "  - Leakage report: academic_outputs/diagnostics/leakage_check_report.md"
        echo "  - Reviewer summary: academic_outputs/diagnostics/mcc0_audit_reviewer_summary.md"
    else
        log_warning "Comprehensive MCC=0 audit failed. Check log: $COMPREHENSIVE_LOG"
    fi

    # Run comprehensive audit
    log_info "Running comprehensive MCC=0 audit..."
    AUDIT_CMD="python audit_mcc_zero.py"
    AUDIT_LOG_FILE="$LOG_DIR/audit_mcc_zero.log"

    if $AUDIT_CMD > "$AUDIT_LOG_FILE" 2>&1; then
        log_success "MCC=0 audit completed"
        echo "  - Audit report: academic_outputs/diagnostics/mcc_zero_audit_report.md"
        echo "  - Audit summary: academic_outputs/diagnostics/mcc_zero_audit_summary.csv"
    else
        log_warning "MCC=0 audit failed. Check log: $AUDIT_LOG_FILE"
    fi
fi

if [ "$DRY_RUN" = false ] && ! has_script "diagnose_mcc_zero.py"; then
    log_info "Skipping MCC=0 diagnostic audit (optional reviewer-only scripts are not included in curated repo)."
fi

# =============================================================================
# Phase 6: Generate Final PDF Report
# =============================================================================

if [ "$DRY_RUN" = false ] && has_script "generate_results_pdf.py"; then
    echo ""
    echo "============================================================================="
    echo "     Phase 6: Generate Final PDF Report"
    echo "============================================================================="
    echo ""

    log_info "Generating comprehensive PDF report..."
    PDF_CMD="python generate_results_pdf.py"
    PDF_LOG_FILE="$LOG_DIR/generate_results_pdf.log"

    if $PDF_CMD > "$PDF_LOG_FILE" 2>&1; then
        log_success "PDF report generated successfully"
        echo "  - PDF report: results.pdf"
        echo "  - Log: $PDF_LOG_FILE"
    else
        log_warning "PDF report generation failed. Check log: $PDF_LOG_FILE"
    fi
fi

if [ "$DRY_RUN" = false ] && ! has_script "generate_results_pdf.py"; then
    log_info "Skipping final PDF generation (optional script not included in curated repo)."
fi

echo ""
echo "============================================================================="
echo "                              Done!"
echo "============================================================================="
