from abc import ABC

import torch
from mmengine.model import BaseModule
from torch import nn

from .base import LightningModule


class MMLabModelAdapter(LightningModule, BaseModule, ABC):
    def __init__(
        self,
        model: nn.Module,
        visualizer_kwargs=None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.model = model

        if visualizer_kwargs is None:
            self.visualizer_kwargs = {}
        else:
            self.visualizer_kwargs = visualizer_kwargs

    def _dump_init_info(self, *args, **kwargs):
        pass

    def set_data_preprocessor_device(self):
        self.model.data_preprocessor.to(self.device)

    def on_fit_start(self):
        self.set_data_preprocessor_device()
        self.init_weights()

    def on_validation_start(self):
        self.set_data_preprocessor_device()

    def on_test_start(self):
        self.set_data_preprocessor_device()

    def on_predict_start(self):
        self.set_data_preprocessor_device()

    def forward(self, batch, mode="loss"):
        self.batch_size = len(batch["inputs"])
        batch = self.model.data_preprocessor(batch, mode != "predict")
        return self.model._run_forward(batch, mode=mode)

    def forward_step(self, batch, *args, split="val", **kwargs):
        outputs = self(batch, mode="predict")
        self.trainer.datamodule.evaluators[split].process(outputs, batch)
        return outputs

    def on_forward_epoch_end(self, *args, split="val", **kwargs):
        log_vars = self.trainer.datamodule.evaluators[split].evaluate(
            len(self.trainer.datamodule.datasets[split])
        )
        self.log_dict(self.flatten_dict(log_vars, split), sync_dist=True)
        return log_vars

    def training_step(self, batch, *args, **kwargs):
        _, log_vars = self.model.parse_losses(self(batch))
        self.log_dict(self.flatten_dict(log_vars))
        return log_vars

    def predict_feature_map_forward(self, inputs, stage="backbone"):
        feature_map_outputs = [
            nn.functional.interpolate(
                output.mean(dim=1, keepdim=True),
                size=inputs.shape[2:],
                mode="bilinear",
            )
            for output in self.model.extract_feat(inputs, stage=stage)
        ]
        feature_map_outputs = [
            ((output - output.min()) / (output.max() - output.min(output)) * 255 + 0.5)
            .clamp_(0, 255)
            .permute(0, 2, 3, 1)
            .to("cpu", torch.uint8)
            .numpy()
            for output in feature_map_outputs
        ]
        return feature_map_outputs

    def predict_forward(self, batch, *args, **kwargs):
        predict_result = {"predict_outputs": self(batch, mode="predict")}

        batch = self.model.data_preprocessor(batch)
        if isinstance(batch, (list, tuple)):
            batch = {"inputs": batch[0], "data_samples": batch[1]}
        predict_result.update(batch)

        predict_result["feature_map_outputs"] = self.predict_feature_map_forward(
            batch["inputs"]
        )
        return predict_result
