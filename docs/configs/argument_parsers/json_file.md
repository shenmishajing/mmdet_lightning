## Introduction ##

A json file parser for lightning CLI, used by wandb sweep.

## Usage ##

Use `--json-file` flag of lightning CLI to load a json config file with this parser, typically used by wandb sweep. For detail, see [wandb sweep doc](https://docs.wandb.ai/guides/sweeps) and next section.

Two points:
- In parameters section of sweep config file, use `\` to split nested config parameters.
- You may want to manually set `export CUDA_VISIBLE_DEVICES=<device_ids>` before run wandb sweep agents.

## A typically wandb sweep config file ##
```yaml
method: bayes
metric:
    goal: minimize
    name: train/loss
parameters:
    optimizer_config/optimizer/init_args/lr:
        distribution: uniform
        max: 0.0002
        min: 5e-05
    optimizer_config/optimizer/init_args/weight_decay:
        distribution: uniform
        max: 0.0002
        min: 5e-05
program: tools/cli.py
command:
    - ${env}
    - ${interpreter}
    - ${program}
    - "fit"
    - "--config"
    - "configs/runs/faster-rcnn/faster-rcnn_r50_fpn_8xb3-2x_coco.yaml"
    - "--json-file"
    - ${args_json_file}
```
