# Method-Level Dataset

This directory contains the method-level version of the released Go bug prediction dataset.

## Purpose

The files in this directory are used when reproducing experiments at the **method granularity**.
Each row represents a method-level instance linked to a particular file and commit, with method-centric complexity and structure features.

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

- `method_bug_metrics.csv`: method instances labeled as buggy
- `method_non_bug_metrics.csv`: method instances labeled as non-buggy

Example project path:

- [method_data/influxdb](influxdb)

## Example schema

A typical method-level CSV contains columns such as:

- `project`
- `file_path`
- `sha`
- `method_name`
- `cyclomatic_complexity`
- `nloc`
- `token_count`
- `parameter_count`
- `defer_count`
- `channel_count`
- `goroutine_count`
- `error_handling_count`
- `loop_count`

These features emphasize method-level control-flow, size, and Go-specific implementation patterns.

## Key identifiers

Important fields for reproduction are:

- `project`: project name
- `file_path`: file containing the method
- `sha`: commit identifier
- `method_name`: method/function identifier when available

Because the repository uses temporal evaluation, commit-linked chronology remains important even at the method level.

## How the main pipeline uses this directory

The main pipeline in [analiz.py](../analiz.py) expects method-level data in:

- `method_data/{project}/method_bug_metrics.csv`
- `method_data/{project}/method_non_bug_metrics.csv`

Example command:

```bash
python analiz.py --project influxdb --level method --methods all --resampling none
```

## Label semantics

- `method_bug_metrics.csv` contains positive examples
- `method_non_bug_metrics.csv` contains negative examples

The released scripts merge these files internally and derive the final target vector during execution.

## Reproducibility notes

1. Keep directory names and CSV names unchanged.
2. Preserve `sha`, because commit-based grouping is part of the temporal evaluation design.
3. Some rows may contain sparse or empty method identifiers depending on extraction constraints; this is part of the released dataset and should not be silently cleaned if exact reproduction is the goal.
4. If you create derived subsets, document the transformation carefully.

## Related documentation

- [README.md](../README.md)
- [temporal_split_methodology.md](../temporal_split_methodology.md)
- [cv_strategies_methodology.md](../cv_strategies_methodology.md)
