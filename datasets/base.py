import copy
import string
from abc import ABC
from collections.abc import Mapping, Sequence

from lightning.pytorch.cli import instantiate_class
from lightning.pytorch.core.datamodule import (
    LightningDataModule as _LightningDataModule,
)
from sklearn.model_selection import KFold
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Subset

from utils import deep_update


class LightningDataModule(_LightningDataModule):
    def __init__(
        self,
        dataset_cfg: dict,
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

    def _setup_dataset(self, split_name):
        self.datasets[split_name] = self._build_data_set(split_name)

    def setup(self, stage=None):
        self.split_names = self._get_split_names(stage)

        for name in self.split_names:
            self._setup_dataset(name)
        self.dataset = self.datasets[self.split_names[0]]
        self.batch_size = self.dataloader_cfg[self.split_names[0]].get("batch_size", 1)

    def _dataloader(self, split_name, **kwargs):
        kwargs["set_batch_size"] = split_name == self.split_names[0]
        return self._build_data_loader(
            self.datasets[split_name], split=split_name, **kwargs
        )

    def train_dataloader(self):
        return self._dataloader("train")

    def val_dataloader(self):
        return self._dataloader("val")

    def test_dataloader(self):
        return self._dataloader("test")

    def predict_dataloader(self):
        return self._dataloader("predict")

    def _build_data_set(self, split):
        return instantiate_class(tuple(), self.dataset_cfg[split])

    def _build_worker_init_fn(self, worker_init_fn_cfg):
        raise NotImplementedError

    def _build_collate_fn(self, collate_fn_cfg):
        raise NotImplementedError

    def _build_sampler(self, sampler_cfg, dataset):
        return instantiate_class((dataset,), sampler_cfg)

    def _build_batch_sampler(self, batch_sampler_cfg, sampler, batch_size, drop_last):
        return instantiate_class((sampler, batch_size, drop_last), batch_sampler_cfg)

    def _construct_data_loader(self, dataset, split="train", set_batch_size=False):
        kwargs = copy.deepcopy(self.dataloader_cfg.get(split, {}))

        if "worker_init_fn" in kwargs:
            kwargs["worker_init_fn"] = self._build_worker_init_fn(
                kwargs["worker_init_fn"]
            )

        if "collate_fn" in kwargs:
            kwargs["collate_fn"] = self._build_collate_fn(kwargs["collate_fn"])

        if "sampler" in kwargs:
            kwargs["sampler"] = self._build_sampler(kwargs["sampler"], dataset)

        if "batch_sampler" in kwargs:
            if set_batch_size:
                kwargs["batch_size"] = self.batch_size

            kwargs["batch_sampler"] = self._build_batch_sampler(
                kwargs["batch_sampler"],
                kwargs.pop("sampler"),
                kwargs.pop("batch_size", 1),
                kwargs.pop("drop_last", False),
            )
        return DataLoader(dataset, **kwargs)

    def _build_data_loader(self, dataset, split="train", set_batch_size=False):
        if isinstance(dataset, Mapping):
            return {
                key: self._build_data_loader(ds, split, set_batch_size)
                for key, ds in dataset.items()
            }
        elif isinstance(dataset, Sequence):
            return [
                self._build_data_loader(ds, split, set_batch_size) for ds in dataset
            ]
        else:
            return self._construct_data_loader(dataset, split, set_batch_size)


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
        return self._build_data_loader(
            self.folds[split_name], split=split_name, **kwargs
        )

    def train_dataloader(self):
        return self._fold_dataloader("train")

    def val_dataloader(self):
        return self._fold_dataloader("val")
