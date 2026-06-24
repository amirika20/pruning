import torch
import torch.nn as nn
from typing import List, TYPE_CHECKING

if TYPE_CHECKING:
    from config import ModelConfig, TransformerModelConfig


class MLP(nn.Module):
    def __init__(self, hidden_sizes: List[int], input_dim: int = 1, output_dim: int = 1):
        super().__init__()
        layers = []
        in_size = input_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(in_size, h))
            layers.append(nn.ReLU())
            in_size = h
        layers.append(nn.Linear(in_size, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ResidualBlock(nn.Module):
    def __init__(self, in_size: int, out_size: int):
        super().__init__()
        self.branch = nn.Sequential(
            nn.Linear(in_size, out_size),
            nn.ReLU(),
            nn.Linear(out_size, out_size),
        )
        self.shortcut = (
            nn.Linear(in_size, out_size, bias=False)
            if in_size != out_size
            else nn.Identity()
        )
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.branch(x) + self.shortcut(x))


class ResNet(nn.Module):
    def __init__(self, hidden_sizes: List[int], input_dim: int = 1, output_dim: int = 1):
        super().__init__()
        sizes = [input_dim] + hidden_sizes
        self.blocks = nn.ModuleList([
            ResidualBlock(sizes[i], sizes[i + 1])
            for i in range(len(hidden_sizes))
        ])
        self.head = nn.Linear(hidden_sizes[-1], output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return self.head(x)


class CNNBlock(nn.Module):
    """Conv2d (no bias) → BatchNorm2d → ReLU."""
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              padding=kernel_size // 2, bias=False)
        self.bn   = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.conv(x)))


class CNN(nn.Module):
    """Stack of CNNBlocks → global average pool → Linear head.
    Input : [B, input_channels, H, W]
    Output: [B, output_dim]
    hidden_sizes defines the output channels of each conv block."""
    def __init__(self, hidden_sizes: List[int], input_channels: int = 1, output_dim: int = 1):
        super().__init__()
        channels = [input_channels] + hidden_sizes
        self.blocks = nn.ModuleList([
            CNNBlock(channels[i], channels[i + 1])
            for i in range(len(hidden_sizes))
        ])
        self.head = nn.Linear(hidden_sizes[-1], output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        x = x.mean(dim=[2, 3])   # global average pooling → [B, C]
        return self.head(x)


class ResCNNBlock(nn.Module):
    """CNN block with a residual shortcut.

    Branch: Conv2d → BN → ReLU → Conv2d → BN   (intermediate dim is prunable)
    Shortcut: identity or 1×1 Conv2d+BN when channels change
    Output: ReLU(branch + shortcut)

    The prunable dimension is the intermediate channel count (output of branch[0],
    input of branch[3]).  The block's output channels and shortcut are untouched
    during pruning — exactly analogous to the MLP ResNet.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super().__init__()
        pad = kernel_size // 2
        self.branch = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=pad, bias=False),  # [0]
            nn.BatchNorm2d(out_channels),                                                  # [1]
            nn.ReLU(),                                                                     # [2]
            nn.Conv2d(out_channels, out_channels, kernel_size, padding=pad, bias=False),  # [3]
            nn.BatchNorm2d(out_channels),                                                  # [4]
        )
        self.shortcut = (
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
            )
            if in_channels != out_channels else nn.Identity()
        )
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.branch(x) + self.shortcut(x))


class ResCNN(nn.Module):
    """Stack of ResCNNBlocks → global average pool → Linear head.
    Input : [B, input_channels, H, W]
    Output: [B, output_dim]
    hidden_sizes defines the output channels of each residual block."""
    def __init__(self, hidden_sizes: List[int], input_channels: int = 1, output_dim: int = 1):
        super().__init__()
        channels = [input_channels] + hidden_sizes
        self.blocks = nn.ModuleList([
            ResCNNBlock(channels[i], channels[i + 1])
            for i in range(len(hidden_sizes))
        ])
        self.head = nn.Linear(hidden_sizes[-1], output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        x = x.mean(dim=[2, 3])   # global average pooling → [B, C]
        return self.head(x)


def build_model(cfg: "ModelConfig", input_dim: int = 1, output_dim: int = 1) -> nn.Module:
    if cfg.arch == "mlp":
        return MLP(cfg.hidden_sizes, input_dim, output_dim)
    elif cfg.arch == "resnet":
        return ResNet(cfg.hidden_sizes, input_dim, output_dim)
    elif cfg.arch == "cnn":
        return CNN(cfg.hidden_sizes, cfg.input_channels, output_dim)
    elif cfg.arch == "rescnn":
        return ResCNN(cfg.hidden_sizes, cfg.input_channels, output_dim)
    else:
        raise ValueError(f"Unknown arch: {cfg.arch!r}. Choose from: mlp, resnet, cnn, rescnn")


# ── Transformer for modular arithmetic ───────────────────────────────────────

class TransformerFFN(nn.Module):
    """Two-layer feed-forward network inside a transformer block (fc1 → ReLU → fc2)."""
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int):
        super().__init__()
        self.attn  = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn   = TransformerFFN(d_model, d_ff)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + attn_out)
        x = self.norm2(x + self.ffn(x))
        return x


class ModularTransformer(nn.Module):
    """Light transformer for modular arithmetic.

    Input : [B, seq_len] int64 tokens
    Output: [B, n_classes] logits classified from the last token position.
    Vocabulary: {0..p-1} ∪ {op=p, eq=p+1}  →  vocab_size = p+2.
    """
    def __init__(self, vocab_size: int, d_model: int, n_heads: int, d_ff: int,
                 n_layers: int, n_classes: int, seq_len: int):
        super().__init__()
        self.tok_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(seq_len, d_model)
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, n_heads, d_ff)
            for _ in range(n_layers)
        ])
        self.head = nn.Linear(d_model, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T = x.shape
        pos = torch.arange(T, device=x.device).unsqueeze(0)
        h = self.tok_embed(x) + self.pos_embed(pos)
        for layer in self.layers:
            h = layer(h)
        return self.head(h[:, -1, :])   # classify from last ("=") token

    def make_dummy_input(self, device: torch.device) -> torch.Tensor:
        seq_len = self.pos_embed.num_embeddings
        return torch.zeros(1, seq_len, dtype=torch.long, device=device)


def build_transformer(cfg: "TransformerModelConfig", vocab_size: int,
                      n_classes: int, seq_len: int) -> ModularTransformer:
    return ModularTransformer(
        vocab_size=vocab_size,
        d_model=cfg.d_model,
        n_heads=cfg.n_heads,
        d_ff=cfg.d_ff,
        n_layers=cfg.n_layers,
        n_classes=n_classes,
        seq_len=seq_len,
    )
