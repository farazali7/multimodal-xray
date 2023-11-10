from pathlib import Path
import torch
from torchvision.models.resnet import ResNet, Bottleneck
from typing import Any, Optional, Union


class ResNet50Extractor(ResNet):
    def __init__(self, pretrained_weights: Optional[Union[str, Path]] = None, **kwargs: Any) -> None:
        """Wraps around PyTorch's ResNet50 model without final classification layer.

        Args:
            pretrained_weights: Path to checkpoint for pretrained weights
            **kwargs: Normal keyword arguments to PyTorch ResNet model
        """
        block = Bottleneck
        layers = [3, 4, 6, 3]
        super(ResNet50Extractor, self).__init__(block=block, layers=layers, **kwargs)

        if pretrained_weights is not None:
            state_dict = torch.load(pretrained_weights, map_location='cpu')
            # Ignore missing projector states for now
            state_dict_new = {k.replace('encoder.encoder.', ''): v for k, v in state_dict.items() if 'projector' not in k}
            self.load_state_dict(state_dict_new)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x0 = self.conv1(x)
        x0 = self.bn1(x0)
        x0 = self.relu(x0)
        x0 = self.maxpool(x0)

        x1 = self.layer1(x0)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        return x4
