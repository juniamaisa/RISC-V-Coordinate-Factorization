from __future__ import annotations

import torch
from torch import nn


class SeparableBlock(nn.Module):
    def __init__(self, width: int, dilation: int = 1, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(width, width, 3, padding=dilation, dilation=dilation, groups=width, bias=False),
            nn.BatchNorm1d(width),
            nn.GELU(),
            nn.Conv1d(width, width, 1, bias=False),
            nn.BatchNorm1d(width),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class GeometryEncoder(nn.Module):
    """Compact encoder with an explicit penultimate representation.

    The adapter is representation-specific. All subsequent blocks are
    structurally identical. ``mask_aware`` removes padded positions after the
    adapter, after every residual block, and during global pooling.
    """

    def __init__(
        self,
        in_channels: int,
        *,
        width: int = 32,
        latent_dim: int = 64,
        depth: int = 3,
        dropout: float = 0.1,
        byte_input: bool = True,
        mask_aware: bool = False,
    ):
        super().__init__()
        self.byte_input = bool(byte_input)
        self.mask_aware = bool(mask_aware)
        self.in_channels = int(in_channels)
        self.width = int(width)
        self.latent_dim = int(latent_dim)
        self.depth = int(depth)
        self.adapter = nn.Sequential(
            nn.Conv1d(in_channels, width, 1, bias=False),
            nn.BatchNorm1d(width),
            nn.GELU(),
        )
        self.blocks = nn.ModuleList(
            [SeparableBlock(width, dilation=2 ** (i % 3), dropout=dropout) for i in range(depth)]
        )
        self.projector = nn.Sequential(nn.Linear(width, latent_dim), nn.LayerNorm(latent_dim))
        self.classifier = nn.Linear(latent_dim, 1)

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float()
        return x / 255.0 if self.byte_input else x

    @staticmethod
    def _canonical_mask(mask: torch.Tensor | None, x: torch.Tensor) -> torch.Tensor | None:
        if mask is None:
            return None
        m = mask.float()
        if m.ndim == 2:
            m = m.unsqueeze(1)
        if m.ndim != 3 or m.shape[0] != x.shape[0] or m.shape[-1] != x.shape[-1]:
            raise ValueError(f"Mask {tuple(m.shape)} incompatible with input {tuple(x.shape)}")
        if m.shape[1] != 1:
            m = (m.max(dim=1, keepdim=True).values > 0).float()
        return (m > 0).float()

    def _pool(self, h: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        if self.mask_aware:
            if mask is None:
                raise ValueError("mask-aware encoder requires a valid-position mask")
            denom = mask.sum(dim=-1).clamp_min(1.0)
            return (h * mask).sum(dim=-1) / denom
        return h.mean(dim=-1)

    def encode_with_intermediates(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        h = self.normalize(x)
        m = self._canonical_mask(mask, h)
        if self.mask_aware:
            if m is None:
                raise ValueError("mask-aware encoder requires an explicit mask")
            h = h * m
        features: dict[str, torch.Tensor] = {}
        h = self.adapter(h)
        if self.mask_aware:
            h = h * m
        features["adapter"] = self._pool(h, m)
        for i, block in enumerate(self.blocks):
            h = block(h)
            if self.mask_aware:
                h = h * m
            features[f"block_{i + 1}"] = self._pool(h, m)
        pooled = self._pool(h, m)
        z = self.projector(pooled)
        features["latent"] = z
        return z, features

    def encode(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        z, _ = self.encode_with_intermediates(x, mask=mask)
        return z

    def forward(self, x: torch.Tensor, *, mask: torch.Tensor | None = None, return_latent: bool = False):
        z = self.encode(x, mask=mask)
        logits = self.classifier(z).squeeze(-1)
        return (logits, z) if return_latent else logits


def build_model(
    representation: str,
    *,
    input_channels: int | None = None,
    mask_aware: bool = False,
    **kwargs,
) -> GeometryEncoder:
    rep = representation.upper()
    default_channels = {"SC": 1, "4C": 4, "32C": 32}.get(rep)
    if default_channels is None:
        raise ValueError(f"Unsupported representation: {representation}")
    channels = int(input_channels if input_channels is not None else default_channels)
    return GeometryEncoder(
        in_channels=channels,
        byte_input=(rep != "32C"),
        mask_aware=mask_aware,
        **kwargs,
    )


def count_trainable_parameters(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def find_width_for_parameter_budget(
    representation: str,
    target_parameters: int,
    *,
    latent_dim: int,
    depth: int,
    input_channels: int | None = None,
    min_width: int = 4,
    max_width: int = 128,
) -> tuple[int, int]:
    """Find the width whose parameter count is closest to a target budget."""
    best: tuple[int, int] | None = None
    for width in range(int(min_width), int(max_width) + 1):
        model = build_model(
            representation,
            width=width,
            latent_dim=latent_dim,
            depth=depth,
            input_channels=input_channels,
        )
        count = count_trainable_parameters(model)
        if best is None or abs(count - target_parameters) < abs(best[1] - target_parameters):
            best = (width, count)
    assert best is not None
    return best
