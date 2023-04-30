from functools import partial

from lightning.pytorch.cli import instantiate_class
from mmengine.dataset import COLLATE_FUNCTIONS

from datasets.base import LightningDataModule


class MMLabDataSetAdapter(LightningDataModule):
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
        if not collate_fn_cfg:
            collate_fn_cfg = {"type": "pseudo_collate"}
        collate_fn_type = collate_fn_cfg.pop("type")
        return partial(COLLATE_FUNCTIONS.get(collate_fn_type), **collate_fn_cfg)  # type: ignore
