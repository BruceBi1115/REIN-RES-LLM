"""Model variants for the rebuttal experiments. Nothing here edits src/.

install() rebinds trainer.build_delta_v3_model inside the calling process only, so REIN and every
other experiment keep using the original model.

  plain_head   Experiment #2 (Table R6). The three heads are replaced by one horizon-dependent
               residual head of matched capacity. Everything else matches NN1, so only the decomposition changes.
  direct_news  Experiment #3 (Table R13). The submitted REIN with only the news interface
               changed. The scalar knobs are held neutral by the run configuration, and a new head of matched
               capacity maps the regime representation directly to an unbounded horizon-length correction.
               The history-derived heads, soft gate, objective and checkpoint rule are unchanged.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from src.delta_v3 import trainer as _trainer
from src.delta_v3.model import DeltaV3Regressor


def _count(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


class PlainResidualRegressor(DeltaV3Regressor):
    """One head of width 3 * hidden_size replaces the slow, shape and spike heads.

    The trainer's slow, shape and spike losses receive constant zeros, so they contribute no gradient
    and only the losses on the final forecast train this head.
    """

    def __init__(self, cfg):
        super().__init__(cfg)
        del self.slow_head, self.shape_head, self.spike_head
        width = 3 * int(cfg.hidden_size)
        self.residual_head = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.LazyLinear(width),
            nn.GELU(),
            nn.Linear(width, int(cfg.horizon)),
        )
        self._capacity_logged = False

    def _log_capacity(self, ts_tokens: torch.Tensor, ts_summary: torch.Tensor) -> None:
        if self._capacity_logged:
            return
        self._capacity_logged = True
        with torch.random.fork_rng(devices=[]):
            reference = DeltaV3Regressor(self.cfg)
            reference.slow_head(ts_summary.detach().cpu())
            reference.shape_head(ts_tokens.detach().cpu())
            reference.spike_head(ts_tokens.detach().cpu())
        replaced = _count(reference.slow_head) + _count(reference.shape_head) + _count(reference.spike_head)
        head = _count(self.residual_head)
        print(
            f"[REIN_EXT][CAPACITY] variant=plain_head tokens={tuple(ts_tokens.shape[1:])} horizon={self.cfg.horizon} "
            f"head_params={head} replaced_params={replaced} ratio={head / replaced:.4f}",
            flush=True,
        )

    def forward(self, history_z, history_times, base_pred_z, base_hidden, regime_pack, residual_history=None):
        ts_tokens, ts_summary = self.ts_encoder(
            history_z=history_z,
            history_times=history_times,
            base_hidden=base_hidden,
            residual_history=residual_history,
        )
        regime_repr = self.encode_regime(regime_pack)
        modulation = self.modulation_heads(ts_summary, regime_repr, regime_pack["relevance_mass"])
        residual_z = self.residual_head(ts_tokens)
        self._log_capacity(ts_tokens, ts_summary)
        pred_z = base_pred_z + modulation["lambda_delta_final"].unsqueeze(-1) * residual_z
        zeros = torch.zeros_like(residual_z)
        batch = residual_z.size(0)
        return {
            "pred_z": pred_z,
            "residual_hat": pred_z - base_pred_z,
            "hard_gate": torch.ones((batch, 1), dtype=base_pred_z.dtype, device=base_pred_z.device),
            "residual_ts_z": residual_z,
            "slow_ts": torch.zeros((batch,), dtype=residual_z.dtype, device=residual_z.device),
            "shape_ts": zeros,
            "shape_z": zeros,
            "spike_ts": zeros,
            "spike_gate_logits_ts": zeros,
            "spike_gate_logits": zeros,
            "spike_gate": zeros,
            "spike_gate_hard": zeros,
            "lambda_delta_final": modulation["lambda_delta_final"],
            "lambda_ts": modulation["lambda_ts"],
            "lambda_news_delta": modulation["lambda_news_delta"],
            "shape_gain": modulation["shape_gain"],
            "shape_gain_raw": modulation["shape_gain_raw"],
            "spike_bias": modulation["spike_bias"],
            "trust_logit": modulation["trust_delta_raw"],
            "active_mask": modulation["active"],
            "relevance_mass": regime_pack["relevance_mass"].to(torch.float32).view(-1),
        }


class DirectNewsRegressor(DeltaV3Regressor):
    """REIN plus a news head that replaces the three news knobs (Delta lambda, shape gain, spike bias).

    Its width is chosen per horizon so that its parameter count matches those three knobs. The final layer
    starts at zero, so training starts from neutral news influence, as the knobs do. Windows that are not
    news-active receive no correction.
    """

    def __init__(self, cfg):
        super().__init__(cfg)
        heads = self.modulation_heads
        replaced = _count(heads.trust_delta_mlp) + _count(heads.shape_gain_mlp) + _count(heads.spike_prior_mlp)
        hidden, horizon = int(cfg.hidden_size), int(cfg.horizon)
        width = max(1, round((replaced - horizon) / (hidden + 1 + horizon)))
        self.news_head = nn.Sequential(nn.Linear(hidden, width), nn.GELU(), nn.Linear(width, horizon))
        nn.init.zeros_(self.news_head[-1].weight)
        nn.init.zeros_(self.news_head[-1].bias)
        head = _count(self.news_head)
        print(
            f"[REIN_EXT][CAPACITY] variant=direct_news horizon={horizon} width={width} "
            f"head_params={head} replaced_params={replaced} ratio={head / replaced:.4f}",
            flush=True,
        )

    def forward(self, history_z, history_times, base_pred_z, base_hidden, regime_pack, residual_history=None):
        out = super().forward(
            history_z=history_z,
            history_times=history_times,
            base_pred_z=base_pred_z,
            base_hidden=base_hidden,
            regime_pack=regime_pack,
            residual_history=residual_history,
        )
        regime_repr = self.encode_regime(regime_pack)
        relevance = regime_pack["relevance_mass"].to(regime_repr.dtype).view(-1)
        heads = self.modulation_heads
        active = (relevance > heads.active_mass_threshold).to(regime_repr.dtype)
        soft_gate = torch.sigmoid((relevance - heads.active_mass_threshold) / heads.active_soft_tau)
        correction = (active * soft_gate).unsqueeze(-1) * self.news_head(regime_repr)
        out["pred_z"] = out["pred_z"] + correction
        out["residual_hat"] = out["pred_z"] - base_pred_z
        out["news_correction"] = correction
        return out


VARIANTS = {"rein": None, "plain_head": PlainResidualRegressor, "direct_news": DirectNewsRegressor}


def install(name: str) -> None:
    """Use variant `name` for every delta model built in this process."""
    if name not in VARIANTS:
        raise ValueError(f"unknown variant {name!r}; choose from {sorted(VARIANTS)}")
    cls = VARIANTS[name]
    if cls is not None:
        _trainer.build_delta_v3_model = lambda cfg: cls(cfg=cfg)
    print(f"[REIN_EXT] variant={name}", flush=True)
