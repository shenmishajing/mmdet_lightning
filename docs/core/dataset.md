## Introduction

The base LightningDataModule to inherit.

## Arguments and config

The base LightningDataModule has two arguments named `dataset_cfg` and `dataloader_cfg`, which are all dicts with [deep update](../configs/deep_update.md) feature supported.

The complete config object will look like:

```yaml
# config object(dataset_cfg or dataloader_cfg)
split_info:
    split_format_to: [ str, ... ]
    split_name_map:
        train: train
        val: val
        test: val
        predict: val
    split_prefix: init_args
    split_attr_split_str: '.'
train:
    <user defined config object to init dataset or dataloader>
val:
    <user defined config object to init dataset or dataloader>
test:
    <user defined config object to init dataset or dataloader>
predict:
    <user defined config object to init dataset or dataloader>
```

Obviously, the `split_info` key is optional, and if all `<config object>` under `train`, `val`, `test` and `predict` key are same, you can just use the `<config object>` as the whole config and omit the `train` etc. level.

### Deep update between split

If there are some thing different between each split like `train` and `val`, you can use the [deep update](../configs/deep_update.md) feature to make it without copy the config object many times, the merge order are from `train` to `val` to `test` to `predict`.

For example, a config file as following:

```yaml
train:
    A:
        abc: 1
    B:
        a: d
        b: e
    C:
        A: a
        B: b
        C: c
val:
    A:
        __delete__: true
    B:
        c: c
test:
    B:
        __delete__: b
        a: a
predict:
    C:
        __delete__: [ A, B ]
        C: d
        D: d
```

will get result as:

```yaml
train:
    A:
        abc: 1
    B:
        a: d
        b: e
    C:
        A: a
        B: b
        C: c
val:
    A: {}
    B:
        a: d
        b: e
        c: c
    C:
        A: a
        B: b
        C: c
test:
    A: {}
    B:
        a: a
        c: c
    C:
        A: a
        B: b
        C: c
predict:
    A: {}
    B:
        a: a
        c: c
    C:
        C: d
        D: d
```

### Split attr set

Sometime, we have to set some argument (like the ann file path) for different split according to the split name, but it's not convenient to write the deep update format to set them, so we support the split attr set feature for this. When you use the feature, you have to write the split level in config object, instead of omitting them, and the config object must be a dict.

First, we use the `split_prefix` to navigate in the config object. Then, we use the `split_attr_split_str` to split every str in `split_format_to` use them to navigate to the attr we have to set, then we will set the `$split` part in that attr to value in `split_name_map` according to current split. Note that, the key in `split_prefix` and the last key in `split_format_to` can not exist, but other keys must exist, otherwise an error will be thrown. When the key in `split_prefix` do not exist, we will set it to a dict. By default, `split_name_map` map all keys expect `train` to `val` for ann file path usage, `split_prefix` is equal to `init_args` for lightning CLI instantiate_class arguments format, for more details, see [arguments with class type doc](https://pytorch-lightning.readthedocs.io/en/stable/cli/lightning_cli_advanced_3.html#trainer-callbacks-and-arguments-with-class-type).

For example, a config file as following:

```yaml
# config object(dataset_cfg or dataloader_cfg)
split_info:
    split_format_to:
        -   ann_file
        -   data_prefix.img
    split_name_map:
        train: train
        val: val
        test: val
        predict: val
    split_prefix: init_args
    split_attr_split_str: '.'
train:
    class_path: mmdet.datasets.coco.CocoDataset
    init_args:
        data_root: data/coco
        ann_file: annotations/instances_${split}2017.json
        data_prefix:
            img: ${split}2017
```

will get result as:

```yaml
# config object(dataset_cfg or dataloader_cfg)
train:
    class_path: mmdet.datasets.coco.CocoDataset
    init_args:
        data_root: data/coco
        ann_file: annotations/instances_train2017.json
        data_prefix:
            img: train2017
val:
    class_path: mmdet.datasets.coco.CocoDataset
    init_args:
        data_root: data/coco
        ann_file: annotations/instances_val2017.json
        data_prefix:
            img: val2017
test:
    class_path: mmdet.datasets.coco.CocoDataset
    init_args:
        data_root: data/coco
        ann_file: annotations/instances_val2017.json
        data_prefix:
            img: val2017
predict:
    class_path: mmdet.datasets.coco.CocoDataset
    init_args:
        data_root: data/coco
        ann_file: annotations/instances_val2017.json
        data_prefix:
            img: val2017
```

### dataset config

By default, the dataset config object follow lightning CLI instantiate_class arguments format (for more details, see [arguments with class type doc](https://pytorch-lightning.readthedocs.io/en/stable/cli/lightning_cli_advanced_3.html#trainer-callbacks-and-arguments-with-class-type)), and we use instantiate_class func to init dataset class. If you want to customize this process, override the `_build_dataset` method of the base LightningDataModule class.

### dataloader config

By default, we use the `torch.utils.data.DataLoader` as the dataloader, which is enough for most time. [pytorch lightning](https://pytorch-lightning.readthedocs.io/en/stable/) [replace the sampler](https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html#replace-sampler-ddp) when using [distributed strategy](https://pytorch-lightning.readthedocs.io/en/stable/api_references.html#strategies) (for details, see [code](https://github.com/Lightning-AI/lightning/blob/b9591d91eea20ed0bf9e191cb99bbfce7a2d2ec7/src/lightning/pytorch/trainer/connectors/data_connector.py#L173-L181)), set `shuffle` to `True` when distributed training and set `shuffle` to `False` when distributed `val`, `test` and `predict` (for details, see [use_distributed_sampler](https://github.com/Lightning-AI/lightning/blob/b9591d91eea20ed0bf9e191cb99bbfce7a2d2ec7/src/lightning/pytorch/trainer/trainer.py#L256-L262)), set `drop_last` to `False` when `val`, `test` and `predict` (for details, see [code](https://github.com/Lightning-AI/lightning/blob/b9591d91eea20ed0bf9e191cb99bbfce7a2d2ec7/src/lightning/pytorch/utilities/data.py#L277-L287)), set `worker_init_fn` any time (for details, see [code](https://github.com/Lightning-AI/lightning/blob/b9591d91eea20ed0bf9e191cb99bbfce7a2d2ec7/src/lightning/pytorch/trainer/connectors/data_connector.py#L501)).

Therefore, there is no need to set `sampler`, `shuffle`, `drop_last` and `worker_init_fn` most time. Mostly, you can set all the other params using config file, expect `collate_fn` and `batch_sampler` args.

#### collate_fn

If you have to set the `collate_fn` func, you may need to implement the `_build_collate_fn` method of the base LightningDataModule class, by default, it will be set to None.

#### batch_sampler

If you have to set the `batch_sampler`, by default, we build a `SequentialSampler` or a `RandomSampler` according to your `shuffle` setting, (by default, `SequentialSampler`, but it dosen't matter most time, because [pytorch lightning](https://pytorch-lightning.readthedocs.io/en/stable/) [replace the sampler](https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html#replace-sampler-ddp) when using [distributed strategy](https://pytorch-lightning.readthedocs.io/en/stable/api_references.html#strategies)), and use this `sampler` and `batch_size`, `drop_last` (by default, False) setting in `dataloader_cfg` with lightning CLI instantiate_class func to init the `batch_sampler`. If you can init your customize `batch_sampler` in this way, you only need to write the config file instead of code. Otherwise, if you can init your customize `batch_smapler` with `dataset` and those three params metioned, you only need to override the `_build_batch_sampler` method of the base LightningDataModule class. Otherwise, you will need to override the `_handle_batch_sampler` method of the base LightningDataModule class.
