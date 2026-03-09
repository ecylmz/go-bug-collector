# Commit-Level Dataset

This directory contains the commit-level version of the released Go bug prediction dataset.

## Purpose

The files in this directory are used when reproducing experiments at the **commit granularity**.
Each row represents a commit instance, with engineered features extracted at the commit level and a bug/non-bug label determined by the dataset construction pipeline.

## Directory layout

Each project has its own subdirectory:

- `caddy/`
- `compose/`
- `consul/`
- `fiber/`
- `gin/`
- `gitea/`
- `grafana/`
- `influxdb/`
- `kubernetes/`
- `minio/`
- `nomad/`
- `packer/`
- `rclone/`
- `terraform/`
- `traefik/`
- `vault/`

A `combined/` directory is also present as part of the released data layout.

## Files inside each project directory

Typical files are:

- `bugs.csv`: commit instances labeled as buggy
- `non_bugs.csv`: commit instances labeled as non-buggy
- `bug_prs.json.gz`: compressed project-specific metadata file present in at least some projects

Example project path:

- [commit_data/influxdb](influxdb)

## Example schema

A typical `bugs.csv` file includes columns such as:

- `sha`
- `is_merge`
- `parents_count`
- `modified_files_count`
- `code_churn`
- `max_file_churn`
- `avg_file_churn`
- `deletions`
- `insertions`
- `net_lines`
- `dmm_unit_size`
- `dmm_unit_complexity`
- `dmm_unit_interfacing`
- `total_token_count`
- `total_nloc`
- `total_complexity`
- `total_changed_method_count`
- `commit_message`
- `commit_timestamp`

The exact set of columns is determined by the released dataset and should be inspected directly if you are performing custom analysis.

## Key identifiers

Important fields for reproduction are:

- `sha`: the commit identifier
- `commit_timestamp`: the temporal ordering field used by the experiment pipeline

The public release already includes timestamp information, so standard experiment execution does not require rebuilding these values.

## How the main pipeline uses this directory

The main pipeline in [analiz.py](../analiz.py) expects commit-level data in:

- `commit_data/{project}/bugs.csv`
- `commit_data/{project}/non_bugs.csv`

When you run:

```bash
python analiz.py --project influxdb --level commit --methods all --resampling none
```

this directory is the data source.

## Semantics of the labels

- `bugs.csv` contains positive examples used as the buggy class
- `non_bugs.csv` contains negative examples used as the clean class

During execution, the pipeline adds or reconstructs the target label internally when combining the files.

## Reproducibility notes

1. Keep the repository root unchanged so the scripts can resolve `commit_data/` correctly.
2. Do not rename project folders or CSV files unless you also update the code.
3. Preserve `commit_timestamp`, because temporal split logic depends on it.
4. The optional PR metadata files are stored in compressed gzip form as `bug_prs.json.gz` to keep the repository lighter for public distribution.
5. If you perform your own aggregation or filtering, document it clearly because it may affect comparability with the released experiments.

## Related documentation

- [README.md](../README.md)
- [temporal_split_methodology.md](../temporal_split_methodology.md)
- [cv_strategies_methodology.md](../cv_strategies_methodology.md)
