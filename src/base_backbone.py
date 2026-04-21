import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F


class _SeriesDecomposition(nn.Module):
    def __init__(self, kernel_size: int):
        super().__init__()
        self.kernel_size = max(1, int(kernel_size))
        self.pad = (self.kernel_size - 1) // 2

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        x: (B, L)
        returns: seasonal, trend (both (B, L))
        """
        x1 = x.unsqueeze(1)  # (B,1,L)
        x_pad = F.pad(x1, (self.pad, self.pad), mode="replicate")
        trend = F.avg_pool1d(x_pad, kernel_size=self.kernel_size, stride=1).squeeze(1)  # (B,L)
        seasonal = x - trend
        return seasonal, trend


class DLinearBackbone(nn.Module):
    """
    Lightweight pure-TS backbone in z-space:
      history_z (B, L) -> base_pred_z (B, H)
    """

    def __init__(self, history_len: int, horizon: int, moving_avg: int = 25, dropout: float = 0.0):
        super().__init__()
        self.history_len = int(history_len)
        self.horizon = int(horizon)
        self.decomp = _SeriesDecomposition(int(moving_avg))
        self.linear_seasonal = nn.Linear(self.history_len, self.horizon)
        self.linear_trend = nn.Linear(self.history_len, self.horizon)
        self.dropout = nn.Dropout(float(dropout))

    def forward(self, history_z: torch.Tensor, return_hidden: bool = False):
        seasonal, trend = self.decomp(history_z)
        y = self.linear_seasonal(seasonal) + self.linear_trend(trend)
        pred = self.dropout(y)
        if not return_hidden:
            return pred
        hidden = torch.cat([seasonal, trend], dim=-1)
        return pred, hidden


class NLinearBackbone(nn.Module):
    """
    NLinear: subtract last value, linear project, add back.
      history_z (B, L) -> base_pred_z (B, H)
    """

    def __init__(self, history_len: int, horizon: int, dropout: float = 0.0):
        super().__init__()
        self.history_len = int(history_len)
        self.horizon = int(horizon)
        self.linear = nn.Linear(self.history_len, self.horizon)
        self.dropout = nn.Dropout(float(dropout))

    def forward(self, history_z: torch.Tensor, return_hidden: bool = False):
        last = history_z[:, -1:]
        x = history_z - last
        y = self.linear(x)
        y = self.dropout(y)
        pred = y + last
        if not return_hidden:
            return pred
        return pred, x


class PatchTSTBackbone(nn.Module):
    """
    Small PatchTST: patch embedding + tiny transformer encoder + flatten head.
      history_z (B, L) -> base_pred_z (B, H)
    """

    def __init__(
        self,
        history_len: int,
        horizon: int,
        patch_len: int = 16,
        patch_stride: int = 8,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.history_len = int(history_len)
        self.horizon = int(horizon)
        self.patch_len = int(patch_len)
        self.patch_stride = int(patch_stride)
        self.d_model = int(d_model)

        L, P, S = self.history_len, self.patch_len, self.patch_stride
        if L < P:
            self.pad_len = P - L
        else:
            remainder = (L - P) % S
            self.pad_len = (S - remainder) if remainder != 0 else 0
        L_padded = L + self.pad_len
        self.num_patches = (L_padded - P) // S + 1

        self.patch_embed = nn.Linear(P, self.d_model)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, self.d_model))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=int(n_heads),
            dim_feedforward=self.d_model * 4,
            dropout=float(dropout),
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=int(n_layers))
        self.head = nn.Linear(self.num_patches * self.d_model, self.horizon)
        self.dropout = nn.Dropout(float(dropout))

    def forward(self, history_z: torch.Tensor, return_hidden: bool = False):
        x = history_z
        if self.pad_len > 0:
            pad = x[:, -1:].expand(-1, self.pad_len)
            x = torch.cat([x, pad], dim=-1)
        patches = x.unfold(dimension=-1, size=self.patch_len, step=self.patch_stride)
        h = self.patch_embed(patches) + self.pos_embed
        h = self.encoder(h)
        hidden = h.flatten(1)
        pred = self.head(self.dropout(hidden))
        if not return_hidden:
            return pred
        return pred, hidden


class MLPBackbone(nn.Module):
    """
    Very small fallback backbone:
      history_z (B, L) -> base_pred_z (B, H)
    """

    def __init__(self, history_len: int, horizon: int, hidden_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.history_len = int(history_len)
        self.horizon = int(horizon)
        h = int(hidden_dim)
        p = float(dropout)
        self.fc1 = nn.Linear(self.history_len, h)
        self.fc2 = nn.Linear(h, h)
        self.head = nn.Linear(h, self.horizon)
        self.dropout = nn.Dropout(p)

    def forward(self, history_z: torch.Tensor, return_hidden: bool = False):
        hidden = F.gelu(self.fc1(history_z))
        hidden = self.dropout(hidden)
        hidden = F.gelu(self.fc2(hidden))
        hidden = self.dropout(hidden)
        pred = self.head(hidden)
        if not return_hidden:
            return pred
        return pred, hidden


def build_base_backbone(
    backbone_name: str,
    history_len: int,
    horizon: int,
    hidden_dim: int = 256,
    moving_avg: int = 25,
    dropout: float = 0.0,
    patch_len: int = 16,
    patch_stride: int = 8,
    d_model: int = 64,
    n_heads: int = 4,
    n_layers: int = 2,
) -> nn.Module:
    name = str(backbone_name).lower()
    if name == "dlinear":
        return DLinearBackbone(history_len=history_len, horizon=horizon, moving_avg=moving_avg, dropout=dropout)
    if name == "mlp":
        return MLPBackbone(history_len=history_len, horizon=horizon, hidden_dim=hidden_dim, dropout=dropout)
    if name == "nlinear":
        return NLinearBackbone(history_len=history_len, horizon=horizon, dropout=dropout)
    if name == "patchtst":
        return PatchTSTBackbone(
            history_len=history_len,
            horizon=horizon,
            patch_len=patch_len,
            patch_stride=patch_stride,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            dropout=dropout,
        )
    raise ValueError(f"Unknown base backbone: {backbone_name}")


def save_base_backbone_checkpoint(
    ckpt_dir: str,
    model: nn.Module,
    backbone_name: str,
    history_len: int,
    horizon: int,
    hidden_dim: int,
    moving_avg: int,
    dropout: float,
    optimizer=None,
    scheduler=None,
    epoch: int | None = None,
    global_step: int | None = None,
    global_zstats: dict | None = None,
    patch_len: int = 16,
    patch_stride: int = 8,
    d_model: int = 64,
    n_heads: int = 4,
    n_layers: int = 2,
):
    os.makedirs(ckpt_dir, exist_ok=True)

    meta = {
        "backbone_name": str(backbone_name),
        "history_len": int(history_len),
        "horizon": int(horizon),
        "hidden_dim": int(hidden_dim),
        "moving_avg": int(moving_avg),
        "dropout": float(dropout),
        "patch_len": int(patch_len),
        "patch_stride": int(patch_stride),
        "d_model": int(d_model),
        "n_heads": int(n_heads),
        "n_layers": int(n_layers),
    }
    if isinstance(global_zstats, dict):
        for key in [
            "normalization_mode",
            "center_global",
            "scale_global",
            "mu_global",
            "sigma_global",
            "quantile_low",
            "quantile_high",
            "quantile_low_value",
            "quantile_high_value",
        ]:
            value = global_zstats.get(key, None)
            if value is None:
                continue
            if key == "normalization_mode":
                meta[key] = str(value)
            else:
                meta[key] = float(value)
    with open(os.path.join(ckpt_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    torch.save({"state_dict": model.state_dict()}, os.path.join(ckpt_dir, "base_backbone.pt"))

    if optimizer is not None:
        state = {
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "global_step": global_step,
            "rng_state": torch.get_rng_state(),
        }
        if torch.cuda.is_available():
            state["cuda_rng_state_all"] = torch.cuda.get_rng_state_all()
        if scheduler is not None:
            state["scheduler"] = scheduler.state_dict()
        torch.save(state, os.path.join(ckpt_dir, "trainer_state.pt"))


def load_base_backbone_checkpoint(
    ckpt_dir: str,
    device=None,
    is_trainable: bool = False,
):
    with open(os.path.join(ckpt_dir, "meta.json"), "r", encoding="utf-8") as f:
        meta = json.load(f)

    model = build_base_backbone(
        backbone_name=meta["backbone_name"],
        history_len=int(meta["history_len"]),
        horizon=int(meta["horizon"]),
        hidden_dim=int(meta.get("hidden_dim", 256)),
        moving_avg=int(meta.get("moving_avg", 25)),
        dropout=float(meta.get("dropout", 0.0)),
        patch_len=int(meta.get("patch_len", 16)),
        patch_stride=int(meta.get("patch_stride", 8)),
        d_model=int(meta.get("d_model", 64)),
        n_heads=int(meta.get("n_heads", 4)),
        n_layers=int(meta.get("n_layers", 2)),
    )
    st = torch.load(os.path.join(ckpt_dir, "base_backbone.pt"), map_location="cpu")
    model.load_state_dict(st["state_dict"], strict=True)

    if device is not None:
        model = model.to(device)
    if not is_trainable:
        model.eval()
        for p in model.parameters():
            p.requires_grad = False
    return model, meta
