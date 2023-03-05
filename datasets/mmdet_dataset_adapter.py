from functools import partial
from typing import Sequence

from lightning.fabric.utilities.distributed import _DatasetSamplerWrapper
from lightning.pytorch.cli import instantiate_class
from mmdet.datasets import AspectRatioBatchSampler as _AspectRatioBatchSampler
from mmengine.dataset import COLLATE_FUNCTIONS

from datasets.base import LightningDataModule


class MMDetDataSetAdapter(LightningDataModule):
    def __init__(self, evaluator_cfg: dict, visualizer_cfg: dict, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.evaluator_cfg = self.get_split_config(evaluator_cfg)
        self.visualizer_cfg = self.get_split_config(visualizer_cfg)

        self.evaluators = {}
        self.visualizers = {}

    def setup(self, stage=None):
        super().setup(stage)

        for name in self.split_names:
            self.evaluators[name] = instantiate_class((), self.evaluator_cfg[name])
            self.visualizers[name] = instantiate_class((), self.visualizer_cfg[name])

            if hasattr(self.datasets[name], "metainfo"):
                self.evaluators[name].dataset_meta = self.datasets[name].metainfo
                self.visualizers[name].dataset_meta = self.datasets[name].metainfo

    def _build_collate_fn(self, collate_fn_cfg):
        # The default behavior of `collat_fn` in dataloader is to
        # merge a list of samples to form a mini-batch of Tensor(s).
        # However, in mmengine, if `collate_fn` is not defined in
        # dataloader_cfg, `pseudo_collate` will only convert the list of
        # samples into a dict without stacking the batch tensor.
        collate_fn_type = collate_fn_cfg.pop("type")
        return partial(COLLATE_FUNCTIONS.get(collate_fn_type), **collate_fn_cfg)  # type: ignore


class AspectRatioBatchSampler(_AspectRatioBatchSampler):
    def __iter__(self) -> Sequence[int]:
        for idx in self.sampler:
            if isinstance(self.sampler.dataset, _DatasetSamplerWrapper):
                data_info = self.sampler.dataset._sampler.data_source.get_data_info(idx)
            else:
                data_info = self.sampler.dataset.get_data_info(idx)
            width, height = data_info["width"], data_info["height"]
            bucket_id = 0 if width < height else 1
            bucket = self._aspect_ratio_buckets[bucket_id]
            bucket.append(idx)
            # yield a batch of indices in the same aspect ratio group
            if len(bucket) == self.batch_size:
                yield bucket[:]
                del bucket[:]

        # yield the rest data and reset the bucket
        left_data = self._aspect_ratio_buckets[0] + self._aspect_ratio_buckets[1]
        self._aspect_ratio_buckets = [[] for _ in range(2)]
        while len(left_data) > 0:
            if len(left_data) <= self.batch_size:
                if not self.drop_last:
                    yield left_data[:]
                left_data = []
            else:
                yield left_data[: self.batch_size]
                left_data = left_data[self.batch_size :]
