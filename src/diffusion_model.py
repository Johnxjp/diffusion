import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, embedding_dim: int, theta: int):
        super().__init__()
        self.dims = embedding_dim
        self.theta = theta

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Takes in a 1-D tensor of noise variance with shape (batch size,)"""

        half_dim = self.dims // 2
        emb = math.log(self.theta) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=x.device) * -emb)
        emb = x[:, None] * emb[None, :]

        # Embeddings shape: (batch size, dims)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb


class Downsample(nn.Module):
    def __init__(self, channels: int, use_conv: bool = True):
        super().__init__()
        if use_conv:
            self.down = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1)
        else:
            self.down = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(x)


class Upsample(nn.Module):
    def __init__(self, channels: int, use_conv: bool = True):
        super().__init__()
        if use_conv:
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
                nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
            )
        else:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.up(x)


class SelfAttention(nn.Module):
    def __init__(self, channels: int, num_groups: int):
        super().__init__()
        self.norm = nn.GroupNorm(num_groups, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.out = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        x_norm = self.norm(x)

        # Get queries, keys, values
        qkv = self.qkv(x_norm).reshape(b, 3, c, h * w)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]

        # Compute attention (corrected einsum for 3D tensors)
        attn = torch.einsum("bcl,bcs->bls", q, k) / math.sqrt(c)
        attn = F.softmax(attn, dim=-1)

        # Apply attention to values
        out = torch.einsum("bls,bcs->bcl", attn, v).reshape(b, c, h, w)

        return self.out(out) + x


class ResidualBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_emb_dim: int,
        num_groups: int,
        kernel_size: int = 3,
        dropout: float = 0,
        use_attention: bool = False,
    ):
        super().__init__()
        self.norm1 = nn.GroupNorm(num_groups, in_channels)
        self.act1 = nn.SiLU()

        # Note: padding="same" ensures that the output size matches the input size
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding="same")

        self.time_act = nn.SiLU()
        self.time_mlp = nn.Linear(time_emb_dim, out_channels)

        self.norm2 = nn.GroupNorm(num_groups, out_channels)
        self.act2 = nn.SiLU()
        self.dropout = nn.Dropout(dropout)

        # The residual conv blocks don't change the input channels
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding="same")
        self.residual = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv2d(in_channels, out_channels, 1)
        )
        self.attention = SelfAttention(out_channels, num_groups) if use_attention else nn.Identity()

    def forward(self, x, time_emb):
        # First part
        x_init = x
        x = self.norm1(x)
        x = self.act1(x)
        x = self.conv1(x)

        # Add time embedding
        time_emb = self.time_mlp(self.time_act(time_emb))[
            :, :, None, None
        ]  # Reshape for broadcasting
        x = x + time_emb

        # Second part
        x = self.norm2(x)
        x = self.act2(x)
        x = self.dropout(x)
        x = self.conv2(x)
        x = self.attention(x)

        return x + self.residual(x_init)


class DiffusionUNet(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        base_channels: int,
        image_resolution: int,
        num_res_blocks: int,
        channel_multipliers: list[int],
        attention_resolutions: list[int],
        resample_with_conv: bool,
        time_emb_dim: int,
        dropout: float,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.base_channels = base_channels
        self.image_resolution = image_resolution
        self.num_res_blocks = num_res_blocks
        self.channel_multipliers = channel_multipliers
        self.attention_resolutions = attention_resolutions
        self.resample_with_conv = resample_with_conv
        self.time_emb_dim = time_emb_dim
        self.dropout = dropout

        group_norm_groups = 32
        theta = 10000
        self.time_embedding = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim, theta),
            nn.Linear(time_emb_dim, time_emb_dim * 4),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 4, time_emb_dim),
        )
        self.conv_in = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)
        self.down_blocks = nn.ModuleList()
        self.middle_blocks = None
        self.up_blocks = nn.ModuleList()

        resolution = image_resolution  # image H x W

        # Downblocks
        downchannels = [base_channels]  # store down channels for skip connections

        # Number of resolutions equal to channel multipliers
        in_channels = base_channels
        for i, multiplier in enumerate(channel_multipliers):
            curr_channels = multiplier * base_channels
            for n in range(num_res_blocks):
                self.down_blocks.append(
                    ResidualBlock(
                        in_channels=in_channels,
                        out_channels=curr_channels,
                        time_emb_dim=time_emb_dim,
                        num_groups=group_norm_groups,
                        dropout=dropout,
                        use_attention=(resolution in attention_resolutions),
                    )
                )
                downchannels.append(curr_channels)
                in_channels = curr_channels

            # Don't downsample on final block
            if i != len(channel_multipliers) - 1:
                self.down_blocks.append(Downsample(curr_channels, use_conv=resample_with_conv))
                downchannels.append(curr_channels)
                resolution = resolution // 2

        self.middle_blocks = nn.ModuleList(
            [
                ResidualBlock(
                    curr_channels,
                    curr_channels,
                    time_emb_dim,
                    group_norm_groups,
                    dropout=dropout,
                    use_attention=True,
                ),
                ResidualBlock(
                    curr_channels, curr_channels, time_emb_dim, group_norm_groups, dropout=dropout
                ),
            ]
        )

        # Upblocks
        in_channels = curr_channels
        resolution = image_resolution // (2 ** len(channel_multipliers))
        for i, multiplier in enumerate(reversed(channel_multipliers)):
            curr_channels = multiplier * base_channels
            for _ in range(num_res_blocks + 1):
                skip = downchannels.pop()
                combined_channels = skip + in_channels
                self.up_blocks.append(
                    # Concatenate input with skip connections
                    ResidualBlock(
                        in_channels=combined_channels,
                        out_channels=curr_channels,
                        time_emb_dim=time_emb_dim,
                        num_groups=group_norm_groups,
                        dropout=dropout,
                        use_attention=(resolution in attention_resolutions),
                    )
                )
                in_channels = curr_channels

            # Don't Upsample on final block
            if i != len(channel_multipliers) - 1:
                self.up_blocks.append(Upsample(curr_channels, use_conv=resample_with_conv))
                resolution = resolution * 2

        self.final_group = nn.GroupNorm(group_norm_groups, curr_channels)
        self.final_act = nn.SiLU()
        self.conv_out = nn.Conv2d(curr_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, noise_variances: torch.Tensor) -> torch.Tensor:
        time_embs = self.time_embedding(noise_variances)
        x = self.conv_in(x)

        down_outputs = [x]
        for i, layer in enumerate(self.down_blocks):
            if isinstance(layer, ResidualBlock):
                x = layer(x, time_embs)
            else:
                x = layer(x)

            down_outputs.append(x)

        for i, layer in enumerate(self.middle_blocks):
            x = layer(x, time_embs)

        for i, layer in enumerate(self.up_blocks):
            if isinstance(layer, ResidualBlock):
                skip = down_outputs.pop()
                x = torch.concat((x, skip), dim=1)  # concatenate on channels
                x = layer(x, time_embs)
            else:
                x = layer(x)

        x = self.final_group(x)
        x = self.final_act(x)
        x = self.conv_out(x)
        return x

    def config(self) -> dict:
        """Return model configuration as a dictionary."""
        return {
            "in_channels": self.in_channels,
            "out_channels": self.out_channels,
            "base_channels": self.base_channels,
            "image_resolution": self.image_resolution,
            "num_res_blocks": self.num_res_blocks,
            "channel_multipliers": self.channel_multipliers,
            "attention_resolutions": self.attention_resolutions,
            "resample_with_conv": self.resample_with_conv,
            "time_emb_dim": self.time_emb_dim,
            "dropout": self.dropout,
        }


if __name__ == "__main__":
    # Example usage
    model = DiffusionUNet(
        in_channels=3,
        out_channels=3,
        base_channels=32,
        image_resolution=64,
        num_res_blocks=2,
        channel_multipliers=[1, 2, 4, 8],
        attention_resolutions=[16],
        dropout=0.1,
        resample_with_conv=True,
        time_emb_dim=128,
    )
    x = torch.randn(1, 3, 64, 64)  # Example input
    noise_variances = torch.randn(1)  # Example noise variance
    output = model(x, noise_variances)
    print(output.shape)  # Should be (1, 3, 64, 64)
