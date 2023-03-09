import copy
import string
from abc import ABC
from collections.abc import Mapping, Sequence

from lightning.pytorch.cli import instantiate_class
from lightning.pytorch.core.datamodule import \
    LightningDataModule as _LightningDataModule
from sklearn.model_selection import KFold
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              Subset)

from utils import deep_update


class LightningDataModule(_LightningDataModule):
    def __init__(
        self,
        dataset_cfg: dict = None,
        dataloader_cfg: dict = None,
    ):
        super().__init__()
        self.split_names = ["train", "val", "test", "predict"]
        self.datasets = {}
        self.dataset = None
        self.batch_size = None

        self.dataset_cfg = self.get_split_config(dataset_cfg)
        self.dataloader_cfg = self.get_split_config(dataloader_cfg)

    def get_split_config(self, config):
        if isinstance(config, Mapping):
            if all([config.get(name) is None for name in self.split_names]):
                return {name: copy.deepcopy(config) for name in self.split_names}
            else:
                res = {}
                last_name = None
                for name in self.split_names:
                    if last_name is None:
                        res[name] = copy.deepcopy(config[name])
                    else:
                        res[name] = deep_update(
                            copy.deepcopy(res[last_name]), config.get(name, {})
                        )
                    last_name = name

                if "split_info" in config:
                    config = config["split_info"]

                    if not isinstance(config["split_format_to"], Sequence):
                        config["split_format_to"] = [config["split_format_to"]]

                    split_name_map = {
                        "train": "train",
                        "val": "val",
                        "test": "val",
                        "predict": "val",
                    }
                    split_name_map.update(config.get("split_name_map", {}))
                    config["split_name_map"] = split_name_map

                    config.setdefault("split_attr_split_str", ".")

                    for name in self.split_names:
                        if res[name].get("init_args") is None:
                            res[name]["init_args"] = {}
                        for split_attr in config["split_format_to"]:
                            cur_cfg = res[name]["init_args"]
                            split_attr = split_attr.split(
                                config["split_attr_split_str"]
                            )
                            for s in split_attr[:-1]:
                                cur_cfg = cur_cfg[s]
                            split_attr = split_attr[-1]
                            cur_cfg[split_attr] = string.Template(
                                cur_cfg.get(split_attr, "$split")
                            ).safe_substitute(split=config["split_name_map"][name])
                return res
        else:
            return {
                name: copy.deepcopy(config) if config else {}
                for name in self.split_names
            }

    def _build_dataset(self, split):
        self.datasets[split] = instantiate_class(tuple(), self.dataset_cfg[split])

    def _build_collate_fn(self, collate_fn_cfg):
        raise NotImplementedError

    def _build_batch_sampler(self, batch_sampler_cfg, *args):
        return instantiate_class(args, batch_sampler_cfg)

    def _build_dataloader(self, dataset, split="train", set_batch_size=False):
        kwargs = copy.deepcopy(self.dataloader_cfg.get(split, {}))

        if set_batch_size:
            kwargs["batch_size"] = self.batch_size

        if "collate_fn" in kwargs:
            kwargs["collate_fn"] = self._build_collate_fn(kwargs["collate_fn"])

        if "batch_sampler" in kwargs:
            # pytorch lightning set shuffle to True when distributed training
            # so we may need to handle this when validation or test or predict
            # or using no-distributed strategy like single device strategy
            # and dp strategy, for detail of no-distributed strategy, see
            # is_distributed func of lightning.pytorch.trainer.connectors.accelerator_connector
            if "shuffle" in kwargs:
                shuffle = kwargs.pop("shuffle")
            else:
                shuffle = False

            if shuffle:
                sampler = RandomSampler(dataset)
            else:
                sampler = SequentialSampler(dataset)

            kwargs["batch_sampler"] = self._build_batch_sampler(
                kwargs["batch_sampler"],
                sampler,
                kwargs.pop("batch_size", 1),
                # pytorch lightning set drop_last to False when validation test or predict
                # so we may need to handle this when training
                kwargs.pop("drop_last", False),
            )
        return DataLoader(dataset, **kwargs)

    def _dataloader(self, split, **kwargs):
        return self._build_dataloader(
            self.datasets[split],
            split=split,
            set_batch_size=split == self.split_names[0],
            **kwargs
        )

    def _get_split_names(self, stage=None):
        if self.trainer.overfit_batches > 0:
            split_names = ["train"]
        elif stage is None:
            split_names = self.split_names
        elif stage == "fit":
            split_names = ["train", "val"]
        elif stage == "validate":
            split_names = ["val"]
        else:
            split_names = [stage.lower()]
        return split_names

    def setup(self, stage=None):
        self.split_names = self._get_split_names(stage)

        for name in self.split_names:
            self._build_dataset(name)
        self.dataset = self.datasets[self.split_names[0]]
        self.batch_size = self.dataloader_cfg[self.split_names[0]].get("batch_size", 1)

    def train_dataloader(self):
        return self._dataloader("train")

    def val_dataloader(self):
        return self._dataloader("val")

    def test_dataloader(self):
        return self._dataloader("test")

    def predict_dataloader(self):
        return self._dataloader("predict")


class KFoldLightningDataModule(LightningDataModule, ABC):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_folds = None
        self.folds = {}
        self.splits = []

    def setup(self, stage=None):
        super().setup(stage)
        if "train" in self.split_names or "val" in self.split_names:
            self.setup_folds(2)
            self.setup_fold_index(0)

    def setup_folds(self, num_folds: int) -> None:
        self.num_folds = num_folds
        self.splits = [
            split for split in KFold(num_folds).split(range(len(self.dataset)))
        ]

    def setup_fold_index(self, fold_index: int) -> None:
        for indices, fold_name in zip(self.splits[fold_index], ["train", "val"]):
            self.folds[fold_name] = Subset(self.dataset, indices)

    def _fold_dataloader(self, split_name, **kwargs):
        return self._build_dataloader(
            self.folds[split_name], split=split_name, **kwargs
        )

    def train_dataloader(self):
        return self._fold_dataloader("train")

    def val_dataloader(self):
        return self._fold_dataloader("val")
