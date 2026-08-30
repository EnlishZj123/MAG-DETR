import math
import os
import re
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...core import register


def _make_adapter_block(in_channels: int, out_channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.GELU(),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.GELU(),
    )


@register()
class DINOv3Encoder(nn.Module):
    """DINOv3 ViT encoder that produces multi-scale feature maps.

    This module is intended to replace both the CNN backbone and HybridEncoder.
    Use with `IdentityBackbone`.

        Implementation notes:
        - If `model_name_or_path` is a local `.pth` file, uses the in-repo `dinov3` hub builders
            (e.g. `dinov3.hub.backbones.dinov3_vits16`) and `get_intermediate_layers`.
        - Otherwise, falls back to HuggingFace `transformers.AutoModel` and extracts intermediate
            hidden states.

    Args:
        model_name_or_path: HF model id or local path.
        out_channels: output channels for each pyramid level.
        feature_layers: transformer block indices to sample (0-based).
        feat_strides: output strides, default [8,16,32].
        freeze_backbone: whether to freeze ViT weights.
        freeze_encoder: whether to freeze the entire encoder (ViT + adapters/pyramid layers).
        patch_size: override patch size if not in config.
        drop_path_rate: optional (passed to from_pretrained via config only if supported).
    """

    def __init__(
        self,
        model_name_or_path: str = "ckpts/dinov3_vits16.pth",
        hub_model: Optional[str] = None,
        out_channels: List[int] = (256, 256, 256),
        feature_layers: List[int] = (3, 7, 11),
        feat_strides: List[int] = (8, 16, 32),
        freeze_backbone: bool = True,
        freeze_encoder: bool = False,
        patch_size: Optional[int] = None,
        pyramid_style: str = "layerwise_fused",
    ):
        super().__init__()

        self.model_name_or_path = self._resolve_weight_path(model_name_or_path)
        self.hub_model = hub_model
        self.out_channels = list(out_channels)
        self.feature_layers = list(feature_layers)
        self.feat_strides = list(feat_strides)
        self.freeze_backbone = freeze_backbone
        self.freeze_encoder = freeze_encoder
        self.patch_size = patch_size
        self.pyramid_style = pyramid_style

        self._backend = "hf"

        if len(self.out_channels) != len(self.feat_strides):
            raise ValueError(
                f"out_channels ({len(self.out_channels)}) must match feat_strides ({len(self.feat_strides)})"
            )

        local_pth = (
            isinstance(self.model_name_or_path, str)
            and os.path.isfile(self.model_name_or_path)
            and self.model_name_or_path.lower().endswith((".pth", ".pt"))
        )

        if local_pth:
            self._backend = "dinov3"
            try:
                from dinov3.hub import backbones as dinov3_backbones
            except Exception as e:  # pragma: no cover
                raise ImportError(
                    "Local .pth loading requires the in-repo `dinov3` package (folder `dinov3/`)."
                ) from e

            hub_model = self.hub_model
            if not hub_model:
                # Try to infer from filename, else default to vits16.
                name = os.path.splitext(os.path.basename(self.model_name_or_path))[0]
                if name.startswith("dinov3_"):
                    hub_model = name
                else:
                    m = re.search(r"dinov3_(vit(?:s|b|l|h|7b)\d+\w*)", self.model_name_or_path)
                    hub_model = m.group(0) if m else "dinov3_vits16"

            builder = getattr(dinov3_backbones, hub_model, None)
            if builder is None or not callable(builder):
                raise ValueError(
                    f"Unknown dinov3 hub_model={hub_model!r}. Expected e.g. 'dinov3_vits16', 'dinov3_vitb16', 'dinov3_vitl16'."
                )

            # Build DinoVisionTransformer and load weights from local path.
            # NOTE: dinov3 hub builders can load local weights via file:// which prints a
            # "Downloading" message and copies into torch hub cache. To avoid confusion and
            # unnecessary caching, we always load local .pth/.pt directly here.
            self.vit = builder(pretrained=False)
            ckpt = torch.load(self.model_name_or_path, map_location="cpu")
            if isinstance(ckpt, dict) and "state_dict" in ckpt:
                ckpt = ckpt["state_dict"]
            if isinstance(ckpt, dict):
                # Strip common prefixes.
                new_state = {}
                for k, v in ckpt.items():
                    if k.startswith("module."):
                        k = k[len("module."):]
                    if k.startswith("backbone."):
                        k = k[len("backbone."):]
                    new_state[k] = v
                ckpt = new_state
            self.vit.load_state_dict(ckpt, strict=True)
            embed_dim = getattr(self.vit, "embed_dim", None)
            if embed_dim is None:
                raise ValueError("Unable to infer ViT embed dim from dinov3 model")
            self.embed_dim = int(embed_dim)

            if self.patch_size is None:
                self.patch_size = int(getattr(self.vit, "patch_size", 16))

            # dinov3 handles CLS/storage tokens internally in get_intermediate_layers.
            self.num_register_tokens = 0

        else:
            # Lazy import so repo still imports without transformers installed.
            try:
                from transformers import AutoModel
            except Exception as e:  # pragma: no cover
                raise ImportError(
                    "DINOv3Encoder requires `transformers` when model_name_or_path is not a local .pth/.pt file. "
                    "Install with `pip install transformers`."
                ) from e

            # We need hidden states to grab intermediate layers.
            self.vit = AutoModel.from_pretrained(
                self.model_name_or_path,
                output_hidden_states=True,
                output_attentions=False,
            )

            embed_dim = getattr(self.vit.config, "hidden_size", None)
            if embed_dim is None:
                embed_dim = getattr(self.vit.config, "embed_dim", None)
            if embed_dim is None:
                raise ValueError("Unable to infer ViT embed dim from model config")
            self.embed_dim = int(embed_dim)

            cfg_patch = getattr(self.vit.config, "patch_size", None)
            if self.patch_size is None:
                self.patch_size = int(cfg_patch) if cfg_patch is not None else 16

            # Some DINO variants use register tokens.
            self.num_register_tokens = int(getattr(self.vit.config, "num_register_tokens", 0) or 0)

        c8, c16, c32 = self.out_channels

        if self.pyramid_style not in {"avg", "layerwise", "layerwise_fused"}:
            raise ValueError(
                f"Unsupported pyramid_style={self.pyramid_style!r}, choose from ['avg','layerwise','layerwise_fused']"
            )

        if self.pyramid_style == "avg":
            # Token -> feature adapters (one per sampled layer), all map to a shared mid channel.
            self.adapters = nn.ModuleList(
                [nn.Conv2d(self.embed_dim, c16, kernel_size=1) for _ in self.feature_layers]
            )

            # Build pyramid from a base stride-16 feature with simple convs.
            self.proj_s8 = nn.Conv2d(c16, c8, kernel_size=1)
            self.proj_s16 = nn.Conv2d(c16, c16, kernel_size=1)
            self.down_s32 = nn.Conv2d(c16, c32, kernel_size=3, stride=2, padding=1)

            self.smooth_s8 = nn.Conv2d(c8, c8, kernel_size=3, padding=1)
            self.smooth_s16 = nn.Conv2d(c16, c16, kernel_size=3, padding=1)
            self.smooth_s32 = nn.Conv2d(c32, c32, kernel_size=3, padding=1)

        elif self.pyramid_style == "layerwise":
            # Layer-wise pyramid:
            # - Use early/mid/late transformer blocks as "S3/S4/S5" semantic levels.
            # - All three originate from the same token grid (stride=patch_size), so we
            #   still need up/down sampling to form [8,16,32] strides.
            if len(self.feature_layers) != 3:
                raise ValueError("pyramid_style='layerwise' expects exactly 3 feature_layers (e.g., [3,7,11])")
            if list(self.feat_strides) != [8, 16, 32]:
                raise ValueError("pyramid_style='layerwise' currently expects feat_strides=[8,16,32]")

            # Each sampled layer gets its own adapter.
            self.adapter_s8 = nn.Conv2d(self.embed_dim, c8, kernel_size=1)
            self.adapter_s16 = nn.Conv2d(self.embed_dim, c16, kernel_size=1)
            self.adapter_s32 = nn.Conv2d(self.embed_dim, c32, kernel_size=1)

            # Up/down sampling + smoothing.
            self.smooth_s8 = nn.Conv2d(c8, c8, kernel_size=3, padding=1)
            self.smooth_s16 = nn.Conv2d(c16, c16, kernel_size=3, padding=1)
            self.down_s32 = nn.Conv2d(c32, c32, kernel_size=3, stride=2, padding=1)
            self.smooth_s32 = nn.Conv2d(c32, c32, kernel_size=3, padding=1)

        else:
            # Layer-wise + fusion pyramid:
            # For each output scale, fuse (3,7,11) layers with learnable weights, then resample to [8,16,32].
            if len(self.feature_layers) != 3:
                raise ValueError("pyramid_style='layerwise_fused' expects exactly 3 feature_layers (e.g., [3,7,11])")
            if list(self.feat_strides) != [8, 16, 32]:
                raise ValueError("pyramid_style='layerwise_fused' currently expects feat_strides=[8,16,32]")

            # Adapters per (scale, layer)
            self.adapters_s8 = nn.ModuleList([_make_adapter_block(self.embed_dim, c8) for _ in range(3)])
            self.adapters_s16 = nn.ModuleList([_make_adapter_block(self.embed_dim, c16) for _ in range(3)])
            self.adapters_s32 = nn.ModuleList([_make_adapter_block(self.embed_dim, c32) for _ in range(3)])

            # Learnable fusion weights per scale (softmaxed at runtime)
            self.fuse_w_s8 = nn.Parameter(torch.zeros(3))
            self.fuse_w_s16 = nn.Parameter(torch.zeros(3))
            self.fuse_w_s32 = nn.Parameter(torch.zeros(3))

            # Learnable resampling is more suitable than direct bilinear upsampling
            # when all three bases originate from the same stride-16 token grid.
            self.up_s8 = nn.Sequential(
                nn.ConvTranspose2d(c8, c8, kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(c8),
                nn.GELU(),
            )

            self.smooth_s8 = nn.Conv2d(c8, c8, kernel_size=3, padding=1)
            self.smooth_s16 = nn.Conv2d(c16, c16, kernel_size=3, padding=1)
            self.down_s32 = nn.Sequential(
                nn.Conv2d(c32, c32, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(c32),
                nn.GELU(),
                nn.Conv2d(c32, c32, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(c32),
            )
            self.smooth_s32 = nn.Conv2d(c32, c32, kernel_size=3, padding=1)
            self.detail_s8 = nn.Sequential(
                nn.Conv2d(c8, c8, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(c8),
                nn.GELU(),
            )
            self.alpha_detail_s8 = nn.Parameter(torch.tensor(0.0))

            # Lightweight top-down fusion with learnable residual gates.
            self.lat_32_to_16 = nn.Conv2d(c32, c16, kernel_size=1)
            self.lat_16_to_8 = nn.Conv2d(c16, c8, kernel_size=1)
            self.alpha_32_to_16 = nn.Parameter(torch.tensor(0.0))
            self.alpha_16_to_8 = nn.Parameter(torch.tensor(0.0))

        if self.freeze_backbone:
            for p in self.vit.parameters():
                p.requires_grad = False

        # Optionally freeze the whole encoder (including adapters / pyramid layers).
        if self.freeze_encoder:
            for p in self.parameters():
                p.requires_grad = False

    @staticmethod
    def _resolve_weight_path(path: str) -> str:
        if not isinstance(path, str) or not path:
            return path
        if os.path.isabs(path) or os.path.isfile(path):
            return path
        # Try resolving relative to repo root (parent of `src/`).
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
        cand = os.path.join(repo_root, path)
        return cand if os.path.isfile(cand) else path

    def _tokens_to_map(self, tokens: torch.Tensor, hw: tuple[int, int]) -> torch.Tensor:
        """Convert [B, N, C] tokens to [B, C, H, W]."""
        B, N, C = tokens.shape
        H, W = hw
        if H * W != N:
            # Best-effort reshape if token count doesn't match expected grid.
            side = int(math.sqrt(N))
            if side * side == N:
                H, W = side, side
            else:
                # Fallback to a rectangular grid.
                H = side
                W = max(1, N // max(1, side))
        feat = tokens[:, : H * W, :].transpose(1, 2).contiguous().view(B, C, H, W)
        return feat

    def _strip_prefix_tokens(self, hidden: torch.Tensor, num_patches: int) -> torch.Tensor:
        """Remove CLS/register tokens so remaining equals num_patches when possible."""
        # hidden: [B, T, C]
        T = hidden.shape[1]
        if T == num_patches:
            return hidden

        # Common cases:
        # - 1 CLS + K register tokens + patches
        # - 1 CLS + patches
        # We detect offset and drop it if reasonable.
        offset = T - num_patches
        if 0 < offset <= 16:
            return hidden[:, offset:, :]

        # Otherwise, at least drop CLS.
        if T > 1:
            return hidden[:, 1:, :]
        return hidden

    def forward(self, x: torch.Tensor):
        # Expect x in [B, 3, H, W]
        B, C, H, W = x.shape

        # Ensure divisible by patch size.
        ps = int(self.patch_size)
        new_h = max(ps, (H // ps) * ps)
        new_w = max(ps, (W // ps) * ps)
        if new_h != H or new_w != W:
            x = F.interpolate(x, size=(new_h, new_w), mode="bilinear", align_corners=False)
            H, W = new_h, new_w

        if self._backend == "dinov3":
            # dinov3 returns already-reshaped BCHW maps when reshape=True.
            maps = self.vit.get_intermediate_layers(x, n=self.feature_layers, reshape=True, norm=True)

            def _get_layer_map(layer_idx: int) -> torch.Tensor:
                # Preserve interface used below: pass actual layer index; map to tuple position.
                try:
                    pos = self.feature_layers.index(layer_idx)
                except ValueError as e:
                    raise IndexError(f"Requested layer {layer_idx} not in feature_layers={self.feature_layers}") from e
                return maps[pos]

            h_patch = H // ps
            w_patch = W // ps

        else:
            # HF ViT expects pixel_values
            outputs = self.vit(pixel_values=x)
            hidden_states = outputs.hidden_states

            h_patch = H // ps
            w_patch = W // ps
            num_patches = h_patch * w_patch

            def _get_layer_map(layer_idx: int) -> torch.Tensor:
                if layer_idx < 0 or layer_idx >= len(hidden_states):
                    raise IndexError(
                        f"feature_layers index {layer_idx} out of range for hidden_states len {len(hidden_states)}"
                    )
                hidden = hidden_states[layer_idx]
                tokens = self._strip_prefix_tokens(hidden, num_patches)
                return self._tokens_to_map(tokens, (h_patch, w_patch))

        if self.pyramid_style == "avg":
            sampled_maps = []
            for i, layer_idx in enumerate(self.feature_layers):
                feat = _get_layer_map(layer_idx)
                feat = self.adapters[i](feat)
                sampled_maps.append(feat)

            # Fuse sampled features (simple average).
            base = torch.stack(sampled_maps, dim=0).mean(dim=0)  # [B, C16, H/16, W/16]

            feat_s16 = self.smooth_s16(self.proj_s16(base))
            feat_s8 = F.interpolate(base, scale_factor=2.0, mode="bilinear", align_corners=False)
            feat_s8 = self.smooth_s8(self.proj_s8(feat_s8))
            feat_s32 = self.smooth_s32(self.down_s32(base))

        elif self.pyramid_style == "layerwise":
            # Layer-wise pyramid: (layer3 -> s8), (layer7 -> s16), (layer11 -> s32)
            l_s8, l_s16, l_s32 = self.feature_layers
            base_s8 = self.adapter_s8(_get_layer_map(l_s8))
            base_s16 = self.adapter_s16(_get_layer_map(l_s16))
            base_s32 = self.adapter_s32(_get_layer_map(l_s32))

            # All bases are at stride=patch_size (usually 16). Convert to desired strides.
            feat_s8 = F.interpolate(base_s8, scale_factor=2.0, mode="bilinear", align_corners=False)
            feat_s8 = self.smooth_s8(feat_s8)

            feat_s16 = self.smooth_s16(base_s16)

            feat_s32 = self.down_s32(base_s32)
            feat_s32 = self.smooth_s32(feat_s32)

        else:
            # Layer-wise fused pyramid.
            l0, l1, l2 = self.feature_layers
            maps = [_get_layer_map(l0), _get_layer_map(l1), _get_layer_map(l2)]

            w8 = torch.softmax(self.fuse_w_s8, dim=0)
            w16 = torch.softmax(self.fuse_w_s16, dim=0)
            w32 = torch.softmax(self.fuse_w_s32, dim=0)

            base_s8 = sum(w8[i] * self.adapters_s8[i](maps[i]) for i in range(3))
            base_s16 = sum(w16[i] * self.adapters_s16[i](maps[i]) for i in range(3))
            base_s32 = sum(w32[i] * self.adapters_s32[i](maps[i]) for i in range(3))

            feat_s16 = self.smooth_s16(base_s16)

            feat_s32 = self.down_s32(base_s32)
            feat_s32 = self.smooth_s32(feat_s32)

            feat_s16 = feat_s16 + self.alpha_32_to_16 * self.lat_32_to_16(
                F.interpolate(feat_s32, size=feat_s16.shape[-2:], mode="bilinear", align_corners=False)
            )

            feat_s8 = self.up_s8(base_s8)
            feat_s8 = feat_s8 + self.alpha_16_to_8 * self.lat_16_to_8(
                F.interpolate(feat_s16, size=feat_s8.shape[-2:], mode="bilinear", align_corners=False)
            )
            feat_s8 = self.smooth_s8(feat_s8)
            feat_s8 = feat_s8 + self.alpha_detail_s8 * self.detail_s8(feat_s8)

        return [feat_s8, feat_s16, feat_s32]
