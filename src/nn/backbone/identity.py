"""Simple identity backbone.

Used when the encoder consumes raw images directly (e.g., ViT-based encoders).
"""

import torch.nn as nn

from ...core import register


@register()
class IdentityBackbone(nn.Module):
    """Pass-through backbone.

    RT-DETR's `RTDETR.forward()` calls `backbone(x)` then `encoder(...)`.
    When using a ViT encoder that takes images directly, set backbone to this.
    """

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x
