# go-bug-collector

`go-bug-collector` is the public reproducibility package for a study on bug prediction in Go projects.

## Licensing and citation

This repository uses a split license model:

- source code is licensed under [LICENSE](LICENSE) using Apache-2.0
- datasets and documentation are licensed under [LICENSE-data](LICENSE-data) using CC BY 4.0

This is the closest common academic-friendly setup to your goal:

- Apache-2.0 keeps the code in a standard, widely accepted open-source form
- CC BY 4.0 requires attribution for data and documentation reuse
- [CITATION.cff](CITATION.cff) provides citation metadata for GitHub and downstream users

Important note: standard open-source licenses generally do **not** let you force a specific academic citation format for code reuse. They can require preservation of license and attribution notices, and CC BY 4.0 can require attribution for the released data/docs. For that reason, this repository also includes [NOTICE](NOTICE) and a citation file to make the expectation explicit.

This repository contains:

- the curated datasets used for the released experiments
- the main experiment pipeline
- the statistical analysis and reporting scripts needed to regenerate study outputs
- methodology documents describing temporal splitting and cross-validation design
- provenance scripts that document how timestamps were added during dataset preparation

It is intended to let an external reader reproduce the main workflow from raw committed inputs to regenerated experiment outputs.

## What is included

The repository intentionally keeps the reproducible core:

- [commit_data](commit_data): commit-level datasets
- [file_data](file_data): file-level datasets
- [method_data](method_data): method-level datasets
- [analiz.py](analiz.py): main training and evaluation pipeline
- [run_all_experiments.sh](run_all_experiments.sh): batch runner for large-scale reproduction
- [statistical_analysis.py](statistical_analysis.py): project-level comparative statistics
- [generate_adequacy_table.py](generate_adequacy_table.py): dataset adequacy summary
- [generate_inclusion_matrix.py](generate_inclusion_matrix.py): project inclusion matrix
- [generate_exploratory_table.py](generate_exploratory_table.py): exploratory appendix-style summary
- [generate_resampling_table.py](generate_resampling_table.py): resampling reproducibility summary
- [academic_figures.py](academic_figures.py): regenerated publication-style figures and summary tables
- [generate_feature_importance_table.py](generate_feature_importance_table.py): aggregated feature-importance outputs
- [cv_strategies_methodology.md](cv_strategies_methodology.md): CV strategy explanation
- [temporal_split_methodology.md](temporal_split_methodology.md): temporal split and leakage-prevention explanation

## What is intentionally excluded

This public repository does not version generated artifacts or reviewer-only appendix machinery that is not required for the main reproducibility path.

Examples of outputs that are expected to be generated locally rather than committed:

- `academic_outputs/`
- `results_commit_level/`
- `results_file_level/`
- `results_method_level/`
- `log/`
- `results.pdf`

Those directories are ignored on purpose and can be rebuilt from the committed datasets and scripts.

## Repository structure

### Core execution

- [analiz.py](analiz.py): trains and evaluates models for commit-, file-, and method-level prediction
- [run_all_experiments.sh](run_all_experiments.sh): orchestrates bulk experiment execution and post-processing
- [optuna_tuning.py](optuna_tuning.py): unified Optuna-based hyperparameter tuning utilities used by the main pipeline
- [feature_select.py](feature_select.py): feature selection helper logic

### Data quality and filtering

- [adequacy_thresholds.py](adequacy_thresholds.py): centralized dataset adequacy thresholds
- [adequacy_filter.py](adequacy_filter.py): helper utilities for filtering projects by adequacy category
- [generate_adequacy_table.py](generate_adequacy_table.py): regenerates adequacy metadata consumed by downstream scripts

### Statistical analysis and reporting

- [statistical_analysis.py](statistical_analysis.py): Friedman, Nemenyi, Wilcoxon, effect sizes, and critical-difference style outputs
- [generate_inclusion_matrix.py](generate_inclusion_matrix.py): PRIMARY / EXPLORATORY / INSUFFICIENT project matrix
- [generate_exploratory_table.py](generate_exploratory_table.py): exploratory-project reporting
- [generate_resampling_table.py](generate_resampling_table.py): resampling scenario documentation
- [academic_figures.py](academic_figures.py): figure and table generation from produced results
- [generate_feature_importance_table.py](generate_feature_importance_table.py): top-feature summaries across projects

### Dataset provenance

- [add_timestamps_all_projects.py](add_timestamps_all_projects.py)
- [add_commit_timestamps.py](add_commit_timestamps.py)

These provenance scripts are included for transparency. They document how timestamp information was attached during dataset preparation, but they are not required for standard reproduction because the released datasets already contain the needed timestamp fields.

## Dataset documentation

Each dataset directory has its own documentation:

- [commit_data/README.md](commit_data/README.md)
- [file_data/README.md](file_data/README.md)
- [method_data/README.md](method_data/README.md)

Read those files first if you want to understand the released data layout and schema before running experiments.

## Supported projects

The released data covers 16 Go projects:

- `caddy`
- `compose`
- `consul`
- `fiber`
- `gin`
- `gitea`
- `grafana`
- `influxdb`
- `kubernetes`
- `minio`
- `nomad`
- `packer`
- `rclone`
- `terraform`
- `traefik`
- `vault`

Each of the three granularity directories contains one subdirectory per project, plus a `combined/` directory that is kept as part of the released data layout.

## Reproducibility goals

There are two practical reproduction targets:

1. **Minimal smoke-test reproduction**: run one project at one granularity level and regenerate a small slice of outputs.
2. **Full study reproduction**: run the full batch workflow across levels, CV strategies, resampling strategies, feature configurations, and statistical summarization.

## Environment setup

### Recommended environment

- macOS or Linux
- Python 3.11 or newer recommended
- a fresh virtual environment

### Installation

From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If you prefer a different environment manager, the key requirement is simply that the packages in [requirements.txt](requirements.txt) are installed in the active Python environment.

## Before running experiments

Make sure you are in the repository root and that the three data directories are present:

- [commit_data](commit_data)
- [file_data](file_data)
- [method_data](method_data)

The main pipeline assumes these directories exist at the repository root.

### Important note about compressed PR metadata files

Some optional metadata files under [commit_data](commit_data) are stored in compressed form as `bug_prs.json.gz` to keep the public repository smaller.

If you need to inspect or process those metadata files directly, decompress them before use. For example:

```bash
find commit_data -name 'bug_prs.json.gz' -exec gzip -dk {} \;
```

The main experiment pipeline primarily depends on the released CSV datasets. However, if any custom preprocessing, inspection, or auxiliary script expects plain `bug_prs.json` files, you should unpack them first.

## Quick reproduction path

This is the fastest way to verify that the repository works end to end.

### Step 1: run a single experiment

Example: method-level experiment for `influxdb` without resampling.

```bash
python analiz.py --project influxdb --level method --methods all --resampling none
```

This should create outputs under a directory such as:

- `results_method_level/influxdb/...`

### Step 2: generate adequacy metadata

Some downstream scripts rely on `academic_outputs/dataset_adequacy.csv`.

Generate it with:

```bash
python generate_adequacy_table.py
```

### Step 3: run a statistical/reporting script

For example:

```bash
python statistical_analysis.py --level method --cv-type temporal --feature-set full
```

Optional follow-up:

```bash
python generate_inclusion_matrix.py
python generate_exploratory_table.py
python generate_resampling_table.py
python academic_figures.py
python generate_feature_importance_table.py
```

## Full-study reproduction path

For a broader reproduction, use the batch runner:

```bash
./run_all_experiments.sh --quick
```

The `--quick` mode is recommended first because it exercises the workflow with a smaller configuration set.

After confirming the environment is working, you can run the broader batch configuration:

```bash
./run_all_experiments.sh
```

Depending on hardware, this may take a long time.

## Important execution details

### Temporal integrity

The study is designed around temporal evaluation. The released data already includes `commit_timestamp`, and the pipeline uses this information to preserve chronological ordering.

See:

- [temporal_split_methodology.md](temporal_split_methodology.md)
- [cv_strategies_methodology.md](cv_strategies_methodology.md)

### Adequacy filtering

The analysis distinguishes between projects that are suitable for primary statistical comparisons and those that should only be treated as exploratory.

This logic is implemented through:

- [adequacy_thresholds.py](adequacy_thresholds.py)
- [adequacy_filter.py](adequacy_filter.py)
- [generate_adequacy_table.py](generate_adequacy_table.py)

If you want reproducible downstream statistics, generate adequacy metadata before running the table/analysis scripts that consume it.

### Hyperparameter optimization

The main public repository keeps the tuning utilities used by the pipeline in [optuna_tuning.py](optuna_tuning.py), and the batch runner still supports the optimization phase.

However, reviewer-specific HPO robustness appendix generation was intentionally removed from the public package. The retained code supports the main experiment and reproducibility workflow without that extra appendix layer.

## Output directories you should expect

During reproduction, the following directories may be created locally:

- `results_commit_level/`
- `results_file_level/`
- `results_method_level/`
- `academic_outputs/`
- `log/`

These are generated outputs and are intentionally excluded from version control.

## Suggested reproduction sequence for external users

If you want a clean, deterministic reproduction process, follow this order:

1. Create and activate a fresh Python environment.
2. Install dependencies from [requirements.txt](requirements.txt).
3. Inspect the dataset notes in [commit_data/README.md](commit_data/README.md), [file_data/README.md](file_data/README.md), and [method_data/README.md](method_data/README.md).
4. Run one small experiment with [analiz.py](analiz.py).
5. Generate dataset adequacy metadata with [generate_adequacy_table.py](generate_adequacy_table.py).
6. Run [statistical_analysis.py](statistical_analysis.py) for the desired level / CV type / feature set.
7. Regenerate summary tables using [generate_inclusion_matrix.py](generate_inclusion_matrix.py), [generate_exploratory_table.py](generate_exploratory_table.py), and [generate_resampling_table.py](generate_resampling_table.py).
8. Generate final figures with [academic_figures.py](academic_figures.py) and [generate_feature_importance_table.py](generate_feature_importance_table.py).
9. If desired, run the curated batch workflow via [run_all_experiments.sh](run_all_experiments.sh).

## Example commands

### Single project, commit level

```bash
python analiz.py --project influxdb --level commit --methods all --resampling none
```

### Single project, file level, shuffle CV

```bash
python analiz.py --project influxdb --level file --methods all --resampling smote --shuffle-cv
```

### Batch smoke test

```bash
./run_all_experiments.sh --quick
```

### Batch run for one level only

```bash
./run_all_experiments.sh --level method
```

### Run optimization phase as well

```bash
./run_all_experiments.sh --optimize
```

## Interpreting the released data

At a high level:

- [commit_data](commit_data) stores one row per commit
- [file_data](file_data) stores one row per file instance associated with a commit
- [method_data](method_data) stores one row per method instance associated with a commit

All three granularities are tied together by project structure and commit identifiers, and the temporal fields enable time-aware splitting.

## Known public-scope limitations

This repository is a curated public package, not a dump of every internal or reviewer-only artifact produced during paper iteration.

That means:

- generated outputs are not committed
- some optional appendix-generation utilities are intentionally absent
- the main reproduction target is the published experimental workflow, not every intermediate research-side convenience script

## If you want to audit the methodology first

Start here:

- [cv_strategies_methodology.md](cv_strategies_methodology.md)
- [temporal_split_methodology.md](temporal_split_methodology.md)
- [commit_data/README.md](commit_data/README.md)
- [file_data/README.md](file_data/README.md)
- [method_data/README.md](method_data/README.md)

## Citation and reuse

If you reuse this repository, cite the associated study and clearly report:

- which granularity level you used
- whether you used temporal or shuffle CV
- which resampling strategy you used
- whether you used the optimization phase
- which subset of projects was included in your statistical analysis

## Final note

The main purpose of this repository is reproducibility. If you follow the environment setup, use the committed datasets, and run the pipeline from the repository root, you should be able to regenerate the main outputs without needing any additional private assets.