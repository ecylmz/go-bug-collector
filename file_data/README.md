# File-Level Dataset

This directory contains the file-level version of the released Go bug prediction dataset.

## Purpose

The files in this directory are used when reproducing experiments at the **file granularity**.
Each row represents a file instance associated with a specific commit, together with static and Go-specific metrics extracted for that file.

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

- `file_bug_metrics.csv`: file instances labeled as buggy
- `file_non_bug_metrics.csv`: file instances labeled as non-buggy

Example project path:

- [file_data/influxdb](influxdb)

## Example schema

A typical file-level CSV contains columns such as:

- `project`
- `file_path`
- `sha`
- `nloc`
- `complexity`
- `token_count`
- `method_count`
- `commit_count`
- `authors_count`
- `avg_method_param_count`
- `import_count`
- `cyclo_per_loc`
- `comment_ratio`
- `struct_count`
- `interface_count`
- `loop_count`
- `error_handling_count`
- `goroutine_count`
- `channel_count`
- `defer_count`
- `context_usage_count`
- `json_tag_count`
- `variadic_function_count`
- `pointer_receiver_count`
- `avg_method_complexity`
- `avg_methods_token_count`

These columns combine generic software metrics and Go-oriented structural features.

## Key identifiers

Important fields for reproduction are:

- `project`: project name
- `file_path`: file-level entity identifier
- `sha`: commit identifier tying the file instance to a specific revision

Temporal execution also depends on commit-level chronology. In practice, the pipeline uses commit-linked ordering information when forming temporal splits.

## How the main pipeline uses this directory

The main pipeline in [analiz.py](../analiz.py) expects file-level data in:

- `file_data/{project}/file_bug_metrics.csv`
- `file_data/{project}/file_non_bug_metrics.csv`

Example command:

```bash
python analiz.py --project influxdb --level file --methods all --resampling none
```

## Label semantics

- `file_bug_metrics.csv` contains positive examples
- `file_non_bug_metrics.csv` contains negative examples

The pipeline combines the two files and constructs the final target label during execution.

## Reproducibility notes

1. Keep file names unchanged; they are hard-coded in the released workflow.
2. Preserve project directory names exactly.
3. Do not remove the commit identifier fields, because downstream logic relies on commit-aware grouping and temporal ordering.
4. If you want to run custom feature ablation, record exactly which columns were removed.

## Related documentation

- [README.md](../README.md)
- [temporal_split_methodology.md](../temporal_split_methodology.md)
- [cv_strategies_methodology.md](../cv_strategies_methodology.md)
