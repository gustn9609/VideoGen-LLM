#!/usr/bin/env python3
"""Small Wan motion-token adapter modules.

These modules are intentionally independent from a specific VideoLLM. They take
Wan VAE grid features and produce a fixed number of motion tokens that can be
concatenated to a VideoLLM visual token stream.
"""

from __future__ import annotations

import torch
from torch import nn


class FactorizedMotionResampler(nn.Module):
    """Resample [B, T, H, W, C] grid features into K motion tokens.

    The module first creates a fixed number of per-frame spatial summary tokens,
    then applies temporal attention and projects to the target LLM dimension.
    This preserves spatial structure longer than global pooling, which was
    important in the synthetic low-FPS experiments.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        tokens_per_frame: int = 4,
        output_tokens: int = 16,
        hidden_dim: int = 512,
        num_heads: int = 8,
        num_layers: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if output_tokens <= 0:
            raise ValueError("output_tokens must be positive")
        if tokens_per_frame <= 0:
            raise ValueError("tokens_per_frame must be positive")

        self.tokens_per_frame = tokens_per_frame
        self.output_tokens = output_tokens
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.spatial_queries = nn.Parameter(torch.randn(tokens_per_frame, hidden_dim) * 0.02)
        self.output_queries = nn.Parameter(torch.randn(output_tokens, hidden_dim) * 0.02)
        self.spatial_attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.temporal_encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.output_attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, output_dim)

    def forward(self, grid: torch.Tensor) -> torch.Tensor:
        """Return motion tokens.

        Args:
            grid: Tensor with shape [B, T, H, W, C].

        Returns:
            Tensor with shape [B, output_tokens, output_dim].
        """
        if grid.ndim != 5:
            raise ValueError(f"Expected [B, T, H, W, C], got shape {tuple(grid.shape)}")
        batch, frames, height, width, channels = grid.shape
        patches = grid.reshape(batch * frames, height * width, channels)
        patches = self.input_proj(patches)

        spatial_q = self.spatial_queries.unsqueeze(0).expand(batch * frames, -1, -1)
        frame_tokens, _ = self.spatial_attn(spatial_q, patches, patches, need_weights=False)
        frame_tokens = frame_tokens.reshape(batch, frames * self.tokens_per_frame, -1)
        frame_tokens = self.temporal_encoder(frame_tokens)

        output_q = self.output_queries.unsqueeze(0).expand(batch, -1, -1)
        tokens, _ = self.output_attn(output_q, frame_tokens, frame_tokens, need_weights=False)
        return self.output_proj(self.norm(tokens))


class WanMotionTokenAdapter(nn.Module):
    """Convenience wrapper for Wan-VAE latent grids.

    Input is expected in Diffusers Wan VAE latent layout [B, C, T, H, W].
    """

    def __init__(
        self,
        latent_dim: int = 16,
        llm_dim: int = 4096,
        tokens_per_frame: int = 4,
        output_tokens: int = 16,
        hidden_dim: int = 512,
        num_heads: int = 8,
        num_layers: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.resampler = FactorizedMotionResampler(
            input_dim=latent_dim,
            output_dim=llm_dim,
            tokens_per_frame=tokens_per_frame,
            output_tokens=output_tokens,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
        )

    def forward(self, wan_latents: torch.Tensor) -> torch.Tensor:
        if wan_latents.ndim != 5:
            raise ValueError(f"Expected [B, C, T, H, W], got shape {tuple(wan_latents.shape)}")
        grid = wan_latents.permute(0, 2, 3, 4, 1).contiguous()
        return self.resampler(grid)


def _smoke() -> None:
    adapter = WanMotionTokenAdapter(latent_dim=16, llm_dim=128, output_tokens=16, hidden_dim=64, num_heads=4)
    x = torch.randn(2, 16, 5, 16, 16)
    y = adapter(x)
    print({"input": list(x.shape), "output": list(y.shape), "parameters": sum(p.numel() for p in adapter.parameters())})


if __name__ == "__main__":
    _smoke()
