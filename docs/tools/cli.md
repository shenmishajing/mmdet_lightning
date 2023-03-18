## CLI ##

CLI script is located at `tools/cli.py`, so you can run it with `python tools/cli.py`.

For commands and options, you can get all available options and commands from `python tools/cli.py --help`.

## Train ##

You can start a experiment with command as follow, in which, `gpu_ids` is comma-separated id list or just one int.

```bash
CUDA_VISIBLE_DEVICES=<gpu_ids> python tools/cli.py fit --config configs/runs/path/to/config
```

## Validation Test and Predict ##

```bash
CUDA_VISIBLE_DEVICES=<gpu_ids> python tools/cli.py {validation, test, predict} --config configs/runs/path/to/config
```

## Tune ##

### auto scale batch size ###

Auto find the largest batch size or largest power of two as batch size. You can get more information from [doc](https://lightning.ai/docs/pytorch/latest/advanced/training_tricks.html#batch-size-finder).

To use this feature with this project, uncommit the callback in default_runtime.yaml, and run `fit`, `validate` etc. commands. But, the auto scale batch size do not support distributed strategy, so set strategy to `single_device` and use only one gpu to get the batch size.
```bash
CUDA_VISIBLE_DEVICES=<gpu_ids> python tools/cli.py fit --config configs/runs/path/to/config --trainer.strategy single_device
```
### auto lr finder ###

Auto find the best learning rate for models, currently only support the first optimizer. You can get more information from [doc](https://lightning.ai/docs/pytorch/latest/advanced/training_tricks.html#learning-rate-finder).

To use this feature with this project, uncommit the callback in default_runtime.yaml, and run `fit`, `validate` etc. commands.
```bash
CUDA_VISIBLE_DEVICES=<gpu_ids> python tools/cli.py fit --config configs/runs/path/to/config
```
