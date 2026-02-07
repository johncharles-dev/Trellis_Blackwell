from typing import *
import torch
import torch.nn as nn
from .. import models
from ..utils.vram_manager import get_vram_manager


class Pipeline:
    """
    A base class for pipelines.
    """
    def __init__(
        self,
        models: dict[str, nn.Module] = None,
    ):
        if models is None:
            return
        self.models = models
        for model in self.models.values():
            model.eval()

    @staticmethod
    def from_pretrained(path: str) -> "Pipeline":
        """
        Load a pretrained model.
        """
        import os
        import json
        is_local = os.path.exists(f"{path}/pipeline.json")

        if is_local:
            config_file = f"{path}/pipeline.json"
        else:
            from huggingface_hub import hf_hub_download
            config_file = hf_hub_download(path, "pipeline.json")

        with open(config_file, 'r') as f:
            args = json.load(f)['args']

        _models = {}
        for k, v in args['models'].items():
            try:
                _models[k] = models.from_pretrained(f"{path}/{v}")
            except:
                _models[k] = models.from_pretrained(v)

        new_pipeline = Pipeline(_models)
        new_pipeline._pretrained_args = args
        return new_pipeline

    @property
    def device(self) -> torch.device:
        for model in self.models.values():
            if hasattr(model, 'device'):
                return model.device
            if hasattr(model, 'parameters'):
                try:
                    return next(model.parameters()).device
                except StopIteration:
                    continue
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def to(self, device_or_dtype) -> "Pipeline":
        """
        Move all models to a device or cast to a dtype.
        """
        for model in self.models.values():
            model.to(device_or_dtype)
        return self

    def cuda(self) -> "Pipeline":
        self.to(torch.device("cuda"))
        return self

    def cpu(self) -> "Pipeline":
        self.to(torch.device("cpu"))
        return self

    def to_dtype(self, dtype: torch.dtype) -> "Pipeline":
        """
        Cast all model parameters to the given dtype.
        """
        for model in self.models.values():
            model.to(dtype)
        return self

    # ---- Model swapping (Layer 4) ----

    def _move_models(self, names: List[str], device: str, empty_cache: bool = True):
        """
        Move specified models to device. Used for CPU<->GPU swapping.
        """
        for name in names:
            if name not in self.models:
                continue
            try:
                current = next(self.models[name].parameters()).device
            except StopIteration:
                continue
            target = torch.device(device)
            if current != target:
                self.models[name].to(device)
        if empty_cache and device == 'cpu':
            torch.cuda.empty_cache()

    def _move_all_models_to_cpu(self):
        """
        Move all models to CPU and free CUDA memory.
        """
        self._move_models(list(self.models.keys()), 'cpu', empty_cache=True)
