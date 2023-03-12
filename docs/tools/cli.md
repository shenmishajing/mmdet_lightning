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

Auto find the largest batch size or largest power of two as batch size. You can get more information from [doc](https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html#auto-scale-batch-size).

To use this feature with this project run
```bash
CUDA_VISIBLE_DEVICES=<gpu_ids> python tools/cli.py tune --config configs/runs/path/to/config --trainer.strategy null --trainer.auto_scale_batch_size {binsearch, power}  --method {fit(by default), validate, test, predict}
```
### auto lr finder ###

Auto find the best learning rate for models, currently only support the first optimizer. You can get more information from [doc](https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html#auto-lr-find).

To use this feature with this project run
```bash
CUDA_VISIBLE_DEVICES=<gpu_ids> python tools/model/lr_finder.py --config configs/runs/path/to/config
```
