#!/usr/bin/env python3
from trojanvision.models.imagemodel import _ImageModel, ImageModel

import torchvision.models
from torchvision.models import MNASNet0_5_Weights, MNASNet0_75_Weights, MNASNet1_0_Weights, MNASNet1_3_Weights
import re

import torch
from collections import OrderedDict

# Map model names to TorchVision weights enums
weights_map = {
    'mnasnet0_5': MNASNet0_5_Weights.IMAGENET1K_V1,
    'mnasnet0_75': MNASNet0_75_Weights.IMAGENET1K_V1,
    'mnasnet1_0': MNASNet1_0_Weights.IMAGENET1K_V1,
    'mnasnet1_3': MNASNet1_3_Weights.IMAGENET1K_V1,
}

class _MNASNet(_ImageModel):

    def __init__(self, mnas_alpha: float, weights=None, **kwargs):
        super().__init__(**kwargs)
        # Select correct weights enum if requested
        _model = torchvision.models.MNASNet(mnas_alpha, num_classes=self.num_classes, weights=weights)
        self.features = _model.layers
        self.classifier = _model.classifier

class MNASNet(ImageModel):
    available_models = ['mnasnet', 'mnasnet0_5', 'mnasnet0_75', 'mnasnet1_0', 'mnasnet1_3']
    weights_map = weights_map

    def __init__(self, name: str = 'mnasnet', mnas_alpha: float = 1.0,
                 model: type[_MNASNet] = _MNASNet, weights=None, **kwargs):
        name, self.mnas_alpha = self.parse_name(name, mnas_alpha)
        # Select weights if available
        weights = self.weights_map.get(name, None)
        super().__init__(name=name, mnas_alpha=self.mnas_alpha, model=model, weights=weights, **kwargs)

    @staticmethod
    def parse_name(name: str, mnas_alpha: float = 1.0) -> tuple[str, float]:
        name_list: list[str] = re.findall('[a-zA-Z]+|[\d_.]+', name)
        name = name_list[0]
        if len(name_list) > 1:
            assert len(name_list) == 2
            mnas_alpha = float(name_list[1].replace('_', '.'))
        return f'{name}{mnas_alpha:.1f}'.replace('.', '_'), mnas_alpha

    def get_official_weights(self, **kwargs) -> OrderedDict[str, torch.Tensor]:
        # Use TorchVision weights API
        weights = self.weights_map.get(self.parse_name('mnasnet', self.mnas_alpha)[0], None)
        if weights is None:
            raise ValueError("No official weights available for this MNASNet variant.")
        _model = torchvision.models.MNASNet(self.mnas_alpha, weights=weights)
        _dict = _model.state_dict()
        new_dict = OrderedDict()
        for key, value in _dict.items():
            if key.startswith('layers.'):
                key = 'features.' + key[7:]
            new_dict[key] = value
        return new_dict
