from __future__ import annotations

import csv
import gc
import json
import math
import os
import shutil
from collections import deque

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup

from ..base.common import (
    _coerce_global_zstats,
    _eval_iter,
    _inv_zscore,
    _json_csv_cell,
    _log_prompt_stats_if_available,
    _normalize_delta_val_mode,
    _open_residual_debug_csv,
    _point_loss,
    _sign_match_pct,
    _single_device_map,
    _z_batch_tensors,
    build_delta_batch_inputs,
    dataStatistic,
    evaluate_metrics_backbone,
)
from ..base_backbone import load_base_backbone_checkpoint
from ..delta_news_hooks import reflect_hard_samples
from ..model2 import build_delta_model, load_checkpoint, save_checkpoint
from ..refine.cache import _init_refine_cache, _init_structured_cache, _prewarm_refine_cache, _save_refine_cache, _save_structured_cache
from ..signnet.training import _compose_delta_with_external_sign, _run_external_signnet, _train_external_signnet
from ..utils.batch_utils import _batch_time_seq_for_sample
from ..utils.metrics import skill_score
from ..utils.residual_utils import split_two_stage_epochs
from ..utils.utils import draw_pred_true, record_test_results_csv
from .core import (
    _build_consistency_loss,
    _build_cleaned_residual_targets,
    _build_delta_optimizer,
    _build_delta_targets,
    _build_relative_state_targets,
    _build_soft_sign_targets,
    _build_windowed_sign_targets,
    _fuse_base_and_delta,
    _log_last_residual_eval_diag,
    _match_horizon_shape,
    _masked_binary_accuracy_from_logits,
    _masked_multiclass_accuracy_from_logits,
    _resolve_residual_arch,
    _resolve_delta_residual_mode,
    _resolve_delta_sign_mode,
    _select_metric,
    _use_external_signnet,
)


def _direct_signed_delta_mode(args, delta_model=None) -> bool:
    arch = _resolve_residual_arch(args)
    if delta_model is not None:
        arch = str(getattr(delta_model, "residual_arch", arch) or arch).lower().strip()
    if arch in {"unified", "simple_concat", "base_only_delta"}:
        return True
    return _resolve_delta_sign_mode(args) == "none"


def _build_direction_target_pack(delta_target: torch.Tensor, args) -> tuple[torch.Tensor, torch.Tensor]:
    mode = str(getattr(args, "sign_label_mode", "hard") or "hard").lower().strip()
    target = delta_target.to(torch.float32)
    if mode == "soft":
        loss_target = _build_soft_sign_targets(target, temperature=float(getattr(args, "sign_soft_temperature", 1.0)))
    elif mode == "windowed":
        loss_target = _build_windowed_sign_targets(target, window_size=int(getattr(args, "sign_window_size", 8)))
    else:
        loss_target = (target > 0).to(torch.float32)
    eval_target = (target > 0).to(torch.float32)
    return loss_target, eval_target


def _maybe_log_batch_news_debug(live_logger, args, *, split: str, epoch_idx: int | None, batch_idx: int, sample_debug_records: list[dict] | None):
    if live_logger is None:
        return
    if int(getattr(args, "batch_news_debug_enable", 0)) != 1:
        return
    records = list(sample_debug_records or [])
    if len(records) == 0:
        return
    max_batches = int(getattr(args, "batch_news_debug_max_batches", -1))
    if max_batches > 0 and batch_idx >= max_batches:
        return

    history_starts = [str(rec.get("history_start", "") or "") for rec in records if str(rec.get("history_start", "") or "").strip()]
    history_ends = [str(rec.get("history_end", "") or "") for rec in records if str(rec.get("history_end", "") or "").strip()]
    cand_counts = [int(rec.get("candidate_news_count", 0) or 0) for rec in records]
    hist_news_counts = [int(rec.get("history_range_news_count", 0) or 0) for rec in records]
    selected_news_counts = [int(rec.get("news_count", 0) or 0) for rec in records]
    doc_total = sum(int(rec.get("temporal_text_doc_total", 0) or 0) for rec in records)
    doc_attached = sum(int(rec.get("temporal_text_doc_attached_any_step", 0) or 0) for rec in records)
    step_total = sum(int(rec.get("temporal_text_step_total", 0) or 0) for rec in records)
    step_tokenized = sum(int(rec.get("temporal_text_step_tokenized_count", 0) or 0) for rec in records)
    epoch_label = "na" if epoch_idx is None else str(int(epoch_idx) + 1)

    live_logger.info(
        "[DELTA][BATCH_DEBUG] "
        f"split={split} epoch={epoch_label} batch={batch_idx + 1} "
        f"history_span=[{history_starts[0] if history_starts else ''} -> {history_ends[-1] if history_ends else ''}] "
        f"candidate_news_sum={sum(cand_counts)} "
        f"history_range_news_sum={sum(hist_news_counts)} "
        f"selected_news_sum={sum(selected_news_counts)} "
        f"temporal_docs_attached={doc_attached}/{doc_total} "
        f"temporal_steps_tokenized={step_tokenized}/{step_total}"
    )

def evaluate_metrics_residual(
    base_model,
    delta_model,
    external_signnet: nn.Module | None,
    tokenizer,
    data_loader,
    templates,
    tpl_id,
    args,
    global_zstats,
    news_df,
    policy_name,
    policy_kw,
    device,
    volatility_bin,
    testing: bool = False,
    true_pred_csv_path: str | None = None,
    news_dropout: bool = False,
    filename: str = None,
    api_adapter=None,
    temporal_text_tokenizer=None,
    residual_debug_csv_path: str | None = None,
    residual_debug_split: str | None = None,
):
    """
    Residual evaluation:
      additive mode: final_pred = base_pred + delta_pred
      relative mode: final_pred = base_raw * (1 + relative_delta)
    Returns:
      final_loss, final_mse, final_mae, base_loss, base_mse, base_mae
    """
    if base_model is None:
        raise ValueError("evaluate_metrics_residual requires a trained base backbone model.")
    stats = _coerce_global_zstats(global_zstats, args, required=True)
    mu_global = float(stats["mu_global"])
    sigma_global = float(stats["sigma_global"])

    base_model.eval()
    delta_model.eval()
    if _use_external_signnet(args) and external_signnet is None:
        raise ValueError("delta_sign_mode=signnet_binary requires an external_signnet model during evaluation.")
    use_external_sign = _use_external_signnet(args) and (external_signnet is not None)
    relative_mode = _resolve_delta_residual_mode(args) == "relative"
    if use_external_sign:
        external_signnet.eval()

    loss_sum, n_samples = 0.0, 0
    base_loss_sum = 0.0
    se_sum, ae_sum, n_elems = 0.0, 0.0, 0
    base_se_sum, base_ae_sum = 0.0, 0.0
    sign_correct_sum = 0.0
    sign_valid_sum = 0.0
    mag_pred_sum = 0.0
    mag_true_sum = 0.0
    delta_abs_sum = 0.0
    sample_final_abs = []
    sample_base_abs = []
    sample_news_count = []
    sample_news_max_utility = []
    sample_base_residual_abs = []
    sample_delta_helped = []
    sample_direction_correct = []
    sample_confidence = []
    sample_route_idx = []
    sample_route_names = tuple(getattr(delta_model, "regime_route_names", ("none", "trend", "event", "reversal", "sparse")))

    if testing:
        ckpt_dir = os.path.join("./checkpoints", args.taskName)
        os.makedirs(ckpt_dir, exist_ok=True)
        ans_json_path = os.path.join(ckpt_dir, f"test_answers_{filename}.json")
    eval_desc = "[EVAL][RESIDUAL][TEST]" if testing else "[EVAL][RESIDUAL][VAL]"
    eval_loader, use_pbar = _eval_iter(data_loader, args, desc=eval_desc)
    debug_fh, debug_writer = _open_residual_debug_csv(residual_debug_csv_path)
    debug_split = str(
        residual_debug_split or ("test" if testing else "val")
    ).strip() or ("test" if testing else "val")
    sample_idx = 0
    try:
        for bidx, batch in enumerate(eval_loader):
            history_z, _, _ = _z_batch_tensors(batch, args, global_zstats=stats)
            history_z = history_z.to(device)
            base_pred = base_model(history_z).to(torch.float32)  # (B,H)
            base_pred_cpu = base_pred.detach().cpu()

            # build delta (with news)
            delta_inputs = build_delta_batch_inputs(
                batch=batch,
                tokenizer=tokenizer,
                temporal_text_tokenizer=temporal_text_tokenizer,
                templates=templates,
                tpl_id=tpl_id,
                args=args,
                global_zstats=stats,
                news_df=news_df,
                policy_name=policy_name,
                policy_kw=policy_kw,
                volatility_bin=volatility_bin,
                testing=testing,
                force_no_news=False,
                news_dropout=news_dropout,
                api_adapter=api_adapter,
            )
            ts_p = delta_inputs["ts_patches"]
            ts_pm = delta_inputs["ts_patch_mask"]
            targets_z = delta_inputs["targets_z"]
            news_counts_d = delta_inputs["news_counts"]
            news_max_utility_d = delta_inputs.get("news_max_utility")
            structured_events_d = delta_inputs["structured_events"]
            structured_doc_events_d = delta_inputs["structured_doc_events"]
            structured_feats_d = delta_inputs["structured_feats"]
            temporal_text_ids_d = delta_inputs.get("temporal_text_ids")
            temporal_text_attn_d = delta_inputs.get("temporal_text_attn")
            temporal_text_step_mask_d = delta_inputs.get("temporal_text_step_mask")
            sample_debug_records = list(delta_inputs.get("sample_debug_records") or [])
            for i_debug, rec in enumerate(sample_debug_records):
                if i_debug < len(news_counts_d):
                    rec["news_count"] = int(float(news_counts_d[i_debug]))
            _maybe_log_batch_news_debug(
                live_logger=getattr(args, "_live_logger", None),
                args=args,
                split=debug_split,
                epoch_idx=None,
                batch_idx=bidx,
                sample_debug_records=sample_debug_records,
            )

            ts_p = ts_p.to(device)
            ts_pm = ts_pm.to(device)
            targets_z = targets_z.to(device)
            structured_feats_d = structured_feats_d.to(device=device, dtype=torch.float32)
            if news_max_utility_d is not None:
                news_max_utility_d = news_max_utility_d.to(device=device, dtype=torch.float32)
            if temporal_text_ids_d is not None:
                temporal_text_ids_d = temporal_text_ids_d.to(device=device, dtype=torch.long)
            if temporal_text_attn_d is not None:
                temporal_text_attn_d = temporal_text_attn_d.to(device=device, dtype=torch.long)
            if temporal_text_step_mask_d is not None:
                temporal_text_step_mask_d = temporal_text_step_mask_d.to(device=device, dtype=torch.long)
            signnet_history_z = history_z.to(torch.float32) if use_external_sign else None

            # delta pred: adapter on + with news
            out_delta = delta_model(
                ts_patches=ts_p,
                ts_patch_mask=ts_pm,
                history_z=history_z,
                base_pred_z=base_pred,
                targets=None,
                head_mode="delta",
                rel_targets=None,
                rel_lambda=0.0,
                structured_feats=structured_feats_d,
                temporal_text_ids=temporal_text_ids_d,
                temporal_text_attn=temporal_text_attn_d,
                temporal_text_step_mask=temporal_text_step_mask_d,
            )
            delta_corr_model = out_delta["pred"].to(torch.float32)
            magnitude = out_delta.get("magnitude", torch.abs(delta_corr_model)).to(torch.float32)
            magnitude_raw = out_delta.get("magnitude_raw", magnitude).to(torch.float32)
            if use_external_sign:
                ctrl_logits, ctrl_score = _run_external_signnet(
                    signnet_model=external_signnet,
                    history_z=signnet_history_z,
                    base_pred_z=base_pred,
                    structured_feats=structured_feats_d,
                    text_summary=out_delta.get("text_summary"),
                    text_strength=out_delta.get("text_strength"),
                    tau=float(getattr(args, "delta_sign_external_tau", 1.0)),
                )
                delta_corr = _compose_delta_with_external_sign(
                    magnitude=magnitude,
                    sign_soft=ctrl_score,
                    delta_clip=float(getattr(delta_model, "delta_clip", getattr(args, "delta_clip", 0.0))),
                    residual_mode=_resolve_delta_residual_mode(args),
                    relative_delta=delta_corr_model,
                )
                if relative_mode:
                    state_logits = ctrl_logits.to(torch.float32)
                    state_score = ctrl_score.to(torch.float32)
                    sign_logits = None
                    sign_soft = None
                else:
                    sign_logits = ctrl_logits.to(torch.float32)
                    sign_soft = ctrl_score.to(torch.float32)
                    state_logits = None
                    state_score = None
            else:
                delta_corr = delta_corr_model
                direct_signed_mode = _direct_signed_delta_mode(args, delta_model)
                if relative_mode and not direct_signed_mode:
                    state_logits = out_delta.get(
                        "state_logits",
                        torch.zeros(delta_corr.size(0), delta_corr.size(1), 3, device=delta_corr.device, dtype=torch.float32),
                    ).to(torch.float32)
                    state_score = out_delta.get("state_score", torch.zeros_like(delta_corr)).to(torch.float32)
                    sign_logits = None
                    sign_soft = None
                else:
                    sign_logits = out_delta.get(
                        "sign_logits",
                        out_delta.get("direction_logits", delta_corr),
                    ).to(torch.float32)
                    sign_soft = out_delta.get(
                        "sign_soft",
                        out_delta.get("direction_score", torch.tanh(sign_logits)),
                    ).to(torch.float32)
                    state_logits = None
                    state_score = None

            targets_cpu = batch["target_value"].detach().cpu().numpy()  # raw

            bs = ts_p.size(0)
            history_z_cpu = history_z.detach().cpu().numpy()
            targets_z_f = targets_z.to(torch.float32)
            targets_z_cpu = targets_z_f.detach().cpu().numpy()
            base_pred_f = base_pred.to(torch.float32)
            delta_corr_f = delta_corr.to(torch.float32)
            magnitude_f = _match_horizon_shape(magnitude.to(torch.float32), delta_corr_f)
            magnitude_raw_f = _match_horizon_shape(magnitude_raw.to(torch.float32), delta_corr_f)
            confidence_f = _match_horizon_shape(
                out_delta.get("confidence", torch.ones_like(delta_corr_f)).to(torch.float32),
                delta_corr_f,
            )
            route_top_idx_tensor = out_delta.get("route_top_idx")
            direct_signed_mode = _direct_signed_delta_mode(args, delta_model)
            if relative_mode and not direct_signed_mode:
                state_logits_f = state_logits.to(torch.float32)
                state_score_f = _match_horizon_shape(state_score.to(torch.float32), delta_corr_f)
                sign_logits_f = None
                sign_soft_f = None
            else:
                sign_logits_f = _match_horizon_shape(sign_logits.to(torch.float32), delta_corr_f)
                sign_soft_f = _match_horizon_shape(sign_soft.to(torch.float32), delta_corr_f)
                state_logits_f = None
                state_score_f = None
            pred_z = _fuse_base_and_delta(
                base_pred_z=base_pred_f,
                delta_pred=delta_corr_f,
                args=args,
                mu_global=mu_global,
                sigma_global=sigma_global,
            )
            loss = F.l1_loss(pred_z, targets_z_f, reduction="mean")
            base_loss = F.l1_loss(base_pred_f, targets_z_f, reduction="mean")
            raw_residual_batch = _build_delta_targets(
                targets_z=targets_z_f,
                base_pred=base_pred_f,
                mu_global=mu_global,
                sigma_global=sigma_global,
                args=args,
            )
            cleaned_residual_batch = _build_cleaned_residual_targets(
                raw_residual=raw_residual_batch,
                structured_feats=structured_feats_d,
                args=args,
            )
            if relative_mode and not direct_signed_mode:
                abs_residual_batch = raw_residual_batch.abs()
                valid_sign_mask = torch.ones_like(abs_residual_batch, dtype=torch.float32)
                state_targets = _build_relative_state_targets(
                    raw_residual_batch,
                    args,
                    structured_feats=structured_feats_d,
                )
                sign_target_bin = None
            else:
                abs_residual_batch = cleaned_residual_batch.abs()
                sign_eps = float(getattr(args, "delta_sign_eps", 0.03))
                valid_sign_mask = (abs_residual_batch > sign_eps).to(torch.float32)
                sign_target_bin = (cleaned_residual_batch > 0).to(torch.float32)
            pred_z_cpu = pred_z.detach().cpu().numpy()
            base_pred_cpu = base_pred_f.detach().cpu().numpy()
            delta_corr_cpu = delta_corr_f.detach().cpu().numpy()
            magnitude_cpu = magnitude_f.detach().cpu().numpy()
            magnitude_raw_cpu = magnitude_raw_f.detach().cpu().numpy()
            if relative_mode and not direct_signed_mode:
                state_logits_cpu = state_logits_f.detach().cpu().numpy()
                state_score_cpu = state_score_f.detach().cpu().numpy()
                sign_logits_cpu = None
                sign_soft_cpu = None
            else:
                sign_logits_cpu = sign_logits_f.detach().cpu().numpy()
                sign_soft_cpu = sign_soft_f.detach().cpu().numpy()
                state_logits_cpu = None
                state_score_cpu = None
            loss_sum += float(loss.detach().cpu()) * bs
            base_loss_sum += float(base_loss.detach().cpu()) * bs
            n_samples += bs
            if relative_mode and not direct_signed_mode:
                sign_correct_sum += float(
                    (_masked_multiclass_accuracy_from_logits(state_logits_f, state_targets, valid_sign_mask) * valid_sign_mask.sum())
                    .detach()
                    .cpu()
                )
            else:
                sign_correct_sum += float(
                    (((sign_logits_f > 0).to(torch.float32) == sign_target_bin).to(torch.float32) * valid_sign_mask)
                    .sum()
                    .detach()
                    .cpu()
                )
            sign_valid_sum += float(valid_sign_mask.sum().detach().cpu())
            mag_pred_sum += float(magnitude_f.sum().detach().cpu())
            mag_true_sum += float(abs_residual_batch.sum().detach().cpu())
            delta_abs_sum += float(delta_corr_f.abs().sum().detach().cpu())
            if use_pbar:
                eval_loader.set_postfix(zMAE=f"{loss_sum / max(1, n_samples):.6f}")

            for i in range(bs):
                pred_denorm = _inv_zscore(pred_z_cpu[i].tolist(), mu_global, sigma_global)
                base_denorm = _inv_zscore(base_pred_cpu[i].tolist(), mu_global, sigma_global)
                true_vals = targets_cpu[i].reshape(-1).tolist()
                true_vals = [float(x) for x in true_vals[: int(args.horizon)]]

                pred = np.asarray(pred_denorm, dtype=np.float32)
                base_only = np.asarray(base_denorm, dtype=np.float32)
                true = np.asarray(true_vals, dtype=np.float32)

                se_sum += float(((pred - true) ** 2).sum())
                ae_sum += float(np.abs(pred - true).sum())
                base_se_sum += float(((base_only - true) ** 2).sum())
                base_ae_sum += float(np.abs(base_only - true).sum())
                n_elems += int(args.horizon)
                final_abs_err = float(np.abs(pred - true).mean())
                base_abs_err = float(np.abs(base_only - true).mean())
                base_residual_abs = float(
                    np.abs(np.asarray(targets_z_cpu[i], dtype=np.float32) - np.asarray(base_pred_cpu[i], dtype=np.float32)).mean()
                )
                if relative_mode and not direct_signed_mode:
                    direction_correct = float(
                        (_match_horizon_shape(state_score_f[i : i + 1], delta_corr_f[i : i + 1]) > 0).to(torch.float32)
                        .eq((raw_residual_batch[i : i + 1] > 0).to(torch.float32))
                        .to(torch.float32)
                        .mean()
                        .detach()
                        .cpu()
                    )
                else:
                    direction_correct = float(
                        ((sign_soft_f[i] > 0).to(torch.float32) == sign_target_bin[i].to(torch.float32))
                        .to(torch.float32)
                        .mean()
                        .detach()
                        .cpu()
                    )
                confidence_value = float(confidence_f[i].mean().detach().cpu())
                route_idx_val = -1
                if route_top_idx_tensor is not None and i < int(route_top_idx_tensor.size(0)):
                    route_idx_val = int(route_top_idx_tensor[i].detach().cpu().item())
                news_count_i = int(float(news_counts_d[i])) if i < len(news_counts_d) else 0
                news_max_utility_i = float(news_max_utility_d[i].detach().cpu()) if news_max_utility_d is not None and i < len(news_max_utility_d) else 0.0
                sample_final_abs.append(final_abs_err)
                sample_base_abs.append(base_abs_err)
                sample_news_count.append(news_count_i)
                sample_news_max_utility.append(news_max_utility_i)
                sample_base_residual_abs.append(base_residual_abs)
                sample_delta_helped.append(float(final_abs_err < base_abs_err))
                sample_direction_correct.append(direction_correct)
                sample_confidence.append(confidence_value)
                sample_route_idx.append(route_idx_val)

                if true_pred_csv_path is not None:
                    with open(true_pred_csv_path, "a", newline="") as f:
                        writer = csv.writer(f)
                        writer.writerows(zip(pred_denorm, true_vals))

                if debug_writer is not None:
                    history_times_i = _batch_time_seq_for_sample(batch.get("history_times"), i)
                    target_times_i = _batch_time_seq_for_sample(batch.get("target_times"), i)
                    sample_debug_i = sample_debug_records[i] if i < len(sample_debug_records) else {}
                    true_residual_z = (
                        np.asarray(targets_z_cpu[i], dtype=np.float32)
                        - np.asarray(base_pred_cpu[i], dtype=np.float32)
                    )
                    pred_residual_z = (
                        np.asarray(pred_z_cpu[i], dtype=np.float32)
                        - np.asarray(base_pred_cpu[i], dtype=np.float32)
                    )
                    residual_mode = _resolve_delta_residual_mode(args)
                    sign_match_pct_additive = ""
                    if residual_mode == "additive":
                        sign_match_pct = _sign_match_pct(true_residual_z, pred_residual_z)
                        sign_match_pct_additive = "" if sign_match_pct is None else float(sign_match_pct)
                    series_id = ""
                    if "series_id" in batch and i < len(batch["series_id"]):
                        series_id = str(batch["series_id"][i])
                    target_time = ""
                    if "target_time" in batch and i < len(batch["target_time"]):
                        target_time = str(batch["target_time"][i])
                    debug_writer.writerow(
                        {
                            "split": debug_split,
                            "sample_idx": sample_idx,
                            "series_id": series_id,
                            "target_time": target_time,
                            "history_start": history_times_i[0] if history_times_i else "",
                            "history_end": history_times_i[-1] if history_times_i else "",
                            "target_start": target_times_i[0] if target_times_i else "",
                            "target_end": target_times_i[-1] if target_times_i else "",
                            "history_times": _json_csv_cell(history_times_i),
                            "target_times": _json_csv_cell(target_times_i),
                            "z_input": _json_csv_cell([float(x) for x in history_z_cpu[i].tolist()]),
                            "target_z": _json_csv_cell([float(x) for x in targets_z_cpu[i].tolist()]),
                            "base_pred_z": _json_csv_cell([float(x) for x in base_pred_cpu[i].tolist()]),
                            "true_residual_z": _json_csv_cell([float(x) for x in true_residual_z.tolist()]),
                            "delta_branch_output": _json_csv_cell([float(x) for x in delta_corr_cpu[i].tolist()]),
                            "pred_residual_z": _json_csv_cell([float(x) for x in pred_residual_z.tolist()]),
                            "pred_residual_sign_match_pct_additive": sign_match_pct_additive,
                            "final_pred_z": _json_csv_cell([float(x) for x in pred_z_cpu[i].tolist()]),
                            "sign_logits": "" if sign_logits_cpu is None else _json_csv_cell([float(x) for x in sign_logits_cpu[i].tolist()]),
                            "sign_soft": "" if sign_soft_cpu is None else _json_csv_cell([float(x) for x in sign_soft_cpu[i].tolist()]),
                            "state_logits": "" if state_logits_cpu is None else _json_csv_cell([[float(v) for v in row] for row in state_logits_cpu[i].tolist()]),
                            "state_score": "" if state_score_cpu is None else _json_csv_cell([float(x) for x in state_score_cpu[i].tolist()]),
                            "magnitude": _json_csv_cell([float(x) for x in magnitude_cpu[i].tolist()]),
                            "magnitude_raw": _json_csv_cell([float(x) for x in magnitude_raw_cpu[i].tolist()]),
                            "news_count": news_count_i,
                            "candidate_news_count": int(sample_debug_i.get("candidate_news_count", 0) or 0),
                            "history_range_news_count": int(sample_debug_i.get("history_range_news_count", 0) or 0),
                            "selected_news_in_history_range_count": int(sample_debug_i.get("selected_news_in_history_range_count", 0) or 0),
                            "news_max_utility": news_max_utility_i,
                            "temporal_text_source": str(sample_debug_i.get("temporal_text_source", "disabled") or "disabled"),
                            "temporal_text_doc_total": int(sample_debug_i.get("temporal_text_doc_total", 0) or 0),
                            "temporal_text_doc_nonempty": int(sample_debug_i.get("temporal_text_doc_nonempty", 0) or 0),
                            "temporal_text_doc_attached_any_step": int(sample_debug_i.get("temporal_text_doc_attached_any_step", 0) or 0),
                            "temporal_text_doc_pre_history": int(sample_debug_i.get("temporal_text_doc_pre_history", 0) or 0),
                            "temporal_text_doc_in_history_range": int(sample_debug_i.get("temporal_text_doc_in_history_range", 0) or 0),
                            "temporal_text_doc_post_history_pre_target": int(sample_debug_i.get("temporal_text_doc_post_history_pre_target", 0) or 0),
                            "temporal_text_step_nonempty_count": int(sample_debug_i.get("temporal_text_step_nonempty_count", 0) or 0),
                            "temporal_text_step_tokenized_count": int(sample_debug_i.get("temporal_text_step_tokenized_count", 0) or 0),
                            "temporal_text_step_total": int(sample_debug_i.get("temporal_text_step_total", 0) or 0),
                            "temporal_text_all_nonempty_docs_attached": int(sample_debug_i.get("temporal_text_all_nonempty_docs_attached", 0) or 0),
                            "base_residual_abs": base_residual_abs,
                            "delta_helped": int(final_abs_err < base_abs_err),
                            "direction_correct": direction_correct,
                            "confidence_value": confidence_value,
                            "regime_route": (
                                sample_route_names[route_idx_val]
                                if 0 <= route_idx_val < len(sample_route_names)
                                else ""
                            ),
                            "policy": str(policy_name),
                            "template_id": int(tpl_id),
                        }
                    )
                    sample_idx += 1

                if testing:
                    structured_merged = (
                        dict(structured_events_d[i])
                        if i < len(structured_events_d) and isinstance(structured_events_d[i], dict)
                        else {}
                    )
                    structured_docs = (
                        list(structured_doc_events_d[i])
                        if i < len(structured_doc_events_d) and isinstance(structured_doc_events_d[i], list)
                        else []
                    )
                    record = {
                        "pred_z": [float(x) for x in pred_z_cpu[i].tolist()],
                        "base_pred_z": [float(x) for x in base_pred_cpu[i].tolist()],
                        "pred": [float(x) for x in pred_denorm],
                        "base_pred": [float(x) for x in base_denorm],
                        "true": [float(x) for x in true_vals],
                        "news_count": news_count_i,
                        "news_max_utility": news_max_utility_i,
                        "structured_events": structured_merged,
                        "structured_events_per_doc": structured_docs,
                        "structured_doc_count": int(len(structured_docs)),
                        "structured_doc_nonempty": int(
                            sum(1 for rec in structured_docs if isinstance(rec, dict) and bool(rec.get("has_events", False)))
                        ),
                        "mu_global": mu_global,
                        "sigma_global": sigma_global,
                        "mu": mu_global,
                        "sigma": sigma_global,
                        "policy": str(policy_name),
                        "template_id": int(tpl_id),
                    }
                    with open(ans_json_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    finally:
        if debug_fh is not None:
            debug_fh.close()

    loss_avg = loss_sum / max(1, n_samples)
    mse_avg = se_sum / max(1, n_elems) if n_elems > 0 else float("inf")
    mae_avg = ae_sum / max(1, n_elems) if n_elems > 0 else float("inf")
    base_loss_avg = base_loss_sum / max(1, n_samples)
    base_mse_avg = base_se_sum / max(1, n_elems) if n_elems > 0 else float("inf")
    base_mae_avg = base_ae_sum / max(1, n_elems) if n_elems > 0 else float("inf")
    news_slice_summary = {}
    event_slice_summary = {}
    if len(sample_final_abs) > 0:
        final_abs_arr = np.asarray(sample_final_abs, dtype=np.float32)
        base_abs_arr = np.asarray(sample_base_abs, dtype=np.float32)
        news_count_arr = np.asarray(sample_news_count, dtype=np.int64)
        base_residual_arr = np.asarray(sample_base_residual_abs, dtype=np.float32)
        direction_arr = np.asarray(sample_direction_correct, dtype=np.float32)
        confidence_arr = np.asarray(sample_confidence, dtype=np.float32)
        route_arr = np.asarray(sample_route_idx, dtype=np.int64)

        news_slices = {
            "no_news": news_count_arr == 0,
            "sparse_news": (news_count_arr >= 1) & (news_count_arr <= 3),
            "dense_news": news_count_arr >= 4,
        }
        for slice_name, mask in news_slices.items():
            if bool(mask.any()):
                news_slice_summary[slice_name] = {
                    "n": int(mask.sum()),
                    "final_mae": float(final_abs_arr[mask].mean()),
                    "base_mae": float(base_abs_arr[mask].mean()),
                }

        threshold = float(np.median(base_residual_arr)) if base_residual_arr.size > 0 else 0.0
        event_slices = {
            "event_period": base_residual_arr > threshold,
            "calm_period": base_residual_arr <= threshold,
        }
        for slice_name, mask in event_slices.items():
            if bool(mask.any()):
                event_slice_summary[slice_name] = {
                    "n": int(mask.sum()),
                    "final_mae": float(final_abs_arr[mask].mean()),
                    "base_mae": float(base_abs_arr[mask].mean()),
                }

        route_summary = {}
        valid_route_mask = route_arr >= 0
        if bool(valid_route_mask.any()):
            unique_routes = np.unique(route_arr[valid_route_mask])
            for route_idx in unique_routes.tolist():
                route_mask = route_arr == route_idx
                route_name = sample_route_names[route_idx] if 0 <= route_idx < len(sample_route_names) else str(route_idx)
                route_summary[route_name] = {
                    "count": int(route_mask.sum()),
                    "final_mae": float(final_abs_arr[route_mask].mean()),
                }
        else:
            route_summary = {}
    else:
        direction_arr = np.asarray([], dtype=np.float32)
        confidence_arr = np.asarray([], dtype=np.float32)
        route_summary = {}

    diag = {
        ("state_acc" if relative_mode and not _direct_signed_delta_mode(args, delta_model) else "sign_acc"): sign_correct_sum / max(1.0, sign_valid_sum),
        "pred_mag_mean": mag_pred_sum / max(1, n_elems),
        "true_abs_residual_mean": mag_true_sum / max(1, n_elems),
        "delta_abs_mean": delta_abs_sum / max(1, n_elems),
        "base_mse": base_mse_avg,
        "base_mae": base_mae_avg,
        "skill_score_mse": skill_score(mse_avg, base_mse_avg),
        "skill_score_mae": skill_score(mae_avg, base_mae_avg),
        "delta_helped_rate": float(np.mean(sample_delta_helped)) if len(sample_delta_helped) > 0 else 0.0,
        "direction_acc": float(direction_arr.mean()) if direction_arr.size > 0 else 0.0,
        "mean_confidence": float(confidence_arr.mean()) if confidence_arr.size > 0 else 0.0,
        "news_slices": news_slice_summary,
        "event_slices": event_slice_summary,
        "route_summary": route_summary,
    }
    setattr(
        args,
        "_last_residual_eval_diag",
        diag,
    )
    return loss_avg, mse_avg, mae_avg, base_loss_avg, base_mse_avg, base_mae_avg

def train_delta_stage(args, bundle, best_base_path: str, best_base_metric):
    live_logger = bundle["live_logger"]
    setattr(args, "_live_logger", live_logger)
    device = bundle["device"]
    train_loader = bundle["train_loader"]
    val_loader = bundle["val_loader"]
    test_loader = bundle["test_loader"]
    news_df = bundle["news_df"]
    policy_kw = []
    templates = bundle["templates"]
    patch_len = bundle["patch_len"]
    volatility_bin = bundle["volatility_bin"]
    volatility_bin_val = bundle["volatility_bin_val"]
    volatility_bin_test = bundle["volatility_bin_test"]
    true_pred_csv_path = bundle["true_pred_csv_path"]
    val_residual_debug_csv_path = bundle.get("val_residual_debug_csv_path")
    test_residual_debug_csv_path = bundle.get("test_residual_debug_csv_path")
    global_zstats_bundle = _coerce_global_zstats(bundle.get("global_zstats", None), args, required=True)
    news_api_adapter = bundle.get("news_api_adapter", None)
    _init_refine_cache(args, live_logger=live_logger)
    _init_structured_cache(args, live_logger=live_logger)
    

    train_cfg = {}

    delta_epochs_override = int(getattr(args, "delta_epochs", -1))
    if delta_epochs_override >= 0:
        delta_epochs = delta_epochs_override
    else:
        base_frac = float(getattr(args, "residual_base_frac", 0.3))
        base_epochs, delta_epochs = split_two_stage_epochs(
            total_epochs=int(args.epochs),
            base_frac=base_frac,
            min_base=int(getattr(args, "residual_min_base_epochs", 1)),
            min_delta=int(getattr(args, "residual_min_delta_epochs", 1)),
        )
        if (args.delta_epochs > 0):
            delta_epochs = args.delta_epochs
        else:
            delta_epochs = args.epochs - base_epochs

    live_logger.info("-----------------------------------------------------")
    live_logger.info(f"[DELTA] Training DELTA: epochs={delta_epochs}, base_ckpt={best_base_path}")
    live_logger.info("-----------------------------------------------------")

    base_backbone, base_meta = load_base_backbone_checkpoint(
        best_base_path,
        device=device,
        is_trainable=False,
    )
    live_logger.info(
        f"[DELTA] Loaded base backbone: {base_meta.get('backbone_name')} "
        f"(L(seq_len)={base_meta.get('history_len')}, H(horizon/pred_len)={base_meta.get('horizon')})"
    )
    # mu_global = mean value of train_df; sigma_global = std value of train_df (with optional z-score clipping)
    # z = (x - mu_global) / sigma_global
    global_zstats = _coerce_global_zstats(base_meta, args, required=False)
    if global_zstats is None:
        global_zstats = global_zstats_bundle
        live_logger.info(
            "[DELTA] base checkpoint has no global z-score stats; "
            f"fallback to train_df stats: mu_global={global_zstats['mu_global']:.6f}, "
            f"sigma_global={global_zstats['sigma_global']:.6f}"
        )
    else:
        live_logger.info(
            "[DELTA] using global z-score stats from base checkpoint: "
            f"mu_global={global_zstats['mu_global']:.6f}, "
            f"sigma_global={global_zstats['sigma_global']:.6f}"
        )

    tokenizer, temporal_text_tokenizer, delta_model = build_delta_model(
        base_model=args.base_model,
        tokenizer_id=args.tokenizer,
        horizon=args.horizon,
        history_len=args.history_len,
        patch_dim=patch_len,
        patch_stride=int(getattr(args, "patch_stride", patch_len)),
        patch_dropout=args.patch_dropout,
        head_dropout=args.head_dropout,
        head_mlp=args.head_mlp,
        delta_head_init_std=float(getattr(args, "delta_head_init_std", 0.01)),
        delta_mag_init_bias=float(getattr(args, "delta_mag_init_bias", -2.0)),
        delta_text_gate_init_bias=float(getattr(args, "delta_text_gate_init_bias", -2.0)),
        delta_clip=float(getattr(args, "delta_clip", 3.0)),
        delta_news_tail_tokens=int(getattr(args, "delta_news_tail_tokens", 160)),
        delta_structured_feature_dim=(
            int(getattr(args, "delta_structured_feature_dim", 12))
            if int(getattr(args, "delta_structured_enable", 0)) == 1
            else 0
        ),
        delta_model_variant=str(getattr(args, "delta_model_variant", "tiny_news_ts")),
        tiny_news_hidden_size=int(getattr(args, "tiny_news_hidden_size", 256)),
        delta_alpha_scale=float(getattr(args, "delta_alpha_scale", 0.75)),
        delta_patch_prototypes=int(getattr(args, "delta_patch_prototypes", 0)),
        delta_patch_proto_temp=float(getattr(args, "delta_patch_proto_temp", 1.0)),
        delta_sign_tau=float(getattr(args, "delta_sign_tau", 1.0)),
        delta_residual_mode=_resolve_delta_residual_mode(args),
        delta_sign_mode=_resolve_delta_sign_mode(args),
        delta_mag_max=float(getattr(args, "delta_mag_max", 0.0)),
        doc_candidate_mode=str(getattr(args, "doc_candidate_mode", "beta_only")),
        # encode news text as inputs
        delta_temporal_text_enable=int(getattr(args, "delta_temporal_text_enable", 0)),
        delta_temporal_text_model_id=str(getattr(args, "temporal_text_model_id", "") or ""),
        delta_temporal_text_dim=int(getattr(args, "delta_temporal_text_dim", 8)),
        delta_temporal_text_fuse_lambda=float(getattr(args, "delta_temporal_text_fuse_lambda", 0.5)),
        delta_temporal_text_freeze_encoder=int(getattr(args, "delta_temporal_text_freeze_encoder", 1)),
        delta_temporal_text_unfreeze_last_n=int(getattr(args, "delta_temporal_text_unfreeze_last_n", 0)),
        delta_text_fuse_mode=str(getattr(args, "delta_text_fuse_mode", "gated_add")),
        delta_multimodal_arch=str(getattr(args, "delta_multimodal_arch", "summary_gated")),
        delta_multimodal_fuse_lambda=float(getattr(args, "delta_multimodal_fuse_lambda", 1.0)),
        residual_arch=str(getattr(args, "residual_arch", "current")),
    )
    delta_model.to(device)
    live_logger.info(
        f"[DELTA] model_variant={str(getattr(delta_model, 'model_variant', 'tiny_news_ts')).lower()} "
        f"multimodal_arch={str(getattr(delta_model, 'multimodal_arch', 'summary_gated'))} "
        f"residual_arch={str(getattr(delta_model, 'residual_arch', 'current'))}"
    )
    if _resolve_residual_arch(args) != "current":
        live_logger.warning(
            "[DELTA][CONFIG_NOTICE] "
            f"residual_arch={_resolve_residual_arch(args)} ignores delta_sign_mode={_resolve_delta_sign_mode(args)} "
            "in the forward path; changing delta_sign_mode only changes metadata/experiment naming here."
        )

    # when to run validation set
    delta_val_mode = _normalize_delta_val_mode(getattr(args, "delta_val_mode", "each_epoch"))
    live_logger.info(f"[DELTA] validation mode: {delta_val_mode}")

    # freeze base head
    for p in delta_model.base_head.parameters():
        p.requires_grad = False

    freeze_feature_modules = int(getattr(args, "delta_freeze_feature_modules", 0)) == 1
    if freeze_feature_modules:
        # legacy option: keep feature extractor fixed during delta adaptation
        freeze_modules = [
            "patch_proj",
            "patch_gate",
            "patch_pos",
            "pool_attn",
            "pool_ln",
            "text_ctx_ln",
            "text2q",
            "temporal_text_tower",
            "temporal_text_gate",
            "regime_router",
            "regime_experts",
        ]
        for name in freeze_modules:
            if hasattr(delta_model, name):
                m = getattr(delta_model, name)
                if hasattr(m, "parameters"):
                    for p in m.parameters():
                        p.requires_grad = False
        if hasattr(delta_model, "pool_q"):
            delta_model.pool_q.requires_grad = False
        if hasattr(delta_model, "layer_w"):
            delta_model.layer_w.requires_grad = False
        live_logger.info("[DELTA] feature modules frozen (legacy mode).")
    else:
        live_logger.info("[DELTA] feature modules remain trainable (recommended).")
    if int(getattr(args, "delta_temporal_text_enable", 0)) == 1:
        if str(getattr(args, "delta_multimodal_arch", "summary_gated") or "summary_gated").lower().strip() == "plan_c_mvp":
            live_logger.warning(
                "[DELTA][CONFIG_NOTICE] multimodal_arch=plan_c_mvp skips patch-level temporal-text fusion; "
                "delta_text_fuse_mode and delta_temporal_text_fuse_lambda do not act on the patch branch in this setting."
            )
        if freeze_feature_modules:
            live_logger.warning(
                "[DELTA][CONFIG_NOTICE] delta_freeze_feature_modules=1 freezes temporal_text_tower / temporal_text_gate; "
                "temporal-text source changes may have only limited downstream impact."
            )
        refine_mode = str(getattr(args, "news_refine_mode", "local") or "local").lower().strip()
        if refine_mode != "api":
            live_logger.info(
                "[DELTA][CONFIG_NOTICE] news_refine_mode is not api; temporal_text_source=refined will use locally truncated raw text, "
                "so differences vs raw may be small."
            )

    # ensure delta-specific heads remain trainable
    train_modules = [
        "delta_head",
        "delta_fuse",
        "delta_mag_head",
        "text_mag_head",
        "route_mag_head",
        "confidence_head",
        "direct_residual_head",
        "simple_delta_head",
        "unified_trunk",
        "text_summary_ln",
        "route_summary_ln",
        "rel_head",
        "text_cross_attn",
    ]
    for name in train_modules:
        if hasattr(delta_model, name):
            m = getattr(delta_model, name)
            if hasattr(m, "parameters"):
                for p in m.parameters():
                    p.requires_grad = True
    if hasattr(delta_model, "delta_log_scale"):
        delta_model.delta_log_scale.requires_grad = True

    optim_delta, lr_info = _build_delta_optimizer(delta_model, args, steps_per_epoch=len(train_loader))
    live_logger.info(
        "[DELTA] optimizer groups: "
        f"base_lr={lr_info['base_lr']:.3e}, "
        f"head_lr={lr_info['head_lr']:.3e} (n={lr_info['n_head']}), "
        f"other_lr={lr_info['other_lr']:.3e} (n={lr_info['n_other']}), "
        f"encoder_lr={lr_info['encoder_lr']:.3e} (n={lr_info['n_encoder']}), "
        f"warmup_steps={lr_info['warmup_steps']}"
    )

    total_opt_steps_delta = math.ceil((len(train_loader) * max(1, delta_epochs)) / max(1, args.grad_accum))
    warmup_steps_delta = int(lr_info.get("warmup_steps", 0))
    warmup_steps_delta = min(warmup_steps_delta, max(0, total_opt_steps_delta - 1))

    if args.scheduler == 1:
        scheduler_delta = get_cosine_schedule_with_warmup(
            optim_delta,
            num_warmup_steps=warmup_steps_delta,
            num_training_steps=total_opt_steps_delta,
        )
    else:
        scheduler_delta = None

    allowed_tpl_ids = sorted([t["id"] for t in templates.values()])
    tpl_id = allowed_tpl_ids[0]
    policy_name = args.default_policy
    best_tpl_id = tpl_id
    best_policy_name = policy_name

    _prewarm_refine_cache(
        args=args,
        news_df=news_df,
        train_df=bundle.get("train_df"),
        val_df=bundle.get("val_df"),
        test_df=bundle.get("test_df"),
        tokenizer=tokenizer,
        live_logger=live_logger,
        api_adapter=news_api_adapter,
    )
    # signnet
    # TRAIN EXTERNAL SIGNNET?
    external_signnet_model = _train_external_signnet(
        args=args,
        base_backbone=base_backbone,
        tokenizer=tokenizer,
        temporal_text_tokenizer=temporal_text_tokenizer,
        temporal_text_tower=getattr(delta_model, "temporal_text_tower", None),
        templates=templates,
        tpl_id=tpl_id,
        policy_name=policy_name,
        policy_kw=policy_kw,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        news_df=news_df,
        volatility_bin=volatility_bin,
        volatility_bin_val=volatility_bin_val,
        volatility_bin_test=volatility_bin_test,
        global_zstats=global_zstats,
        device=device,
        live_logger=live_logger,
        api_adapter=news_api_adapter,
    )
    external_sign_enabled = external_signnet_model is not None
    if not external_sign_enabled:
        setattr(
            args,
            "_last_signnet_metrics",
            {
                "test_acc": float("nan"),
                "test_bacc": float("nan"),
                "test_valid": float("nan"),
                "task_type": "disabled",
                "residual_arch": _resolve_residual_arch(args),
            },
        )
    relative_mode = _resolve_delta_residual_mode(args) == "relative"
    if external_sign_enabled:
        live_logger.info(
            "[DELTA] external signnet is active: DELTA internal sign branch is overridden during train/val/test."
        )

    if math.isfinite(float(best_base_metric)):
        base_ref_metric = float(best_base_metric)
        live_logger.info(
            f"[DELTA] base reference on val: threshold={base_ref_metric:.6f} "
            f"(metric={str(args.select_metric).lower()})"
        )
    else:
        base_ref_loss, base_ref_mse, base_ref_mae = evaluate_metrics_backbone(
            base_backbone=base_backbone,
            data_loader=val_loader,
            args=args,
            global_zstats=global_zstats,
            device=device,
            testing=False,
            true_pred_csv_path=None,
            filename=None,
        )
        if args.select_metric == "loss":
            base_ref_metric = float(base_ref_loss)
        elif args.select_metric == "mse":
            base_ref_metric = float(base_ref_mse)
        else:
            base_ref_metric = float(base_ref_mae)
        live_logger.info(
            f"[DELTA] base reference on val: loss={base_ref_loss:.6f} mse={base_ref_mse:.6f} "
            f"mae={base_ref_mae:.6f}; threshold={base_ref_metric:.6f}"
        )
    best_metric = float("inf")
    stale_rounds = 0
    has_saved_delta = False
    loss_window = deque(maxlen=50)

    val_loss_per_epoch, mse_loss_per_epoch, mae_loss_per_epoch = [], [], []
    global_step = 0
    best_delta_alpha = 1.0
    early_stop_patience = max(0, int(getattr(args, "early_stop_patience", 0))) if delta_val_mode == "each_epoch" else 0
    best_delta_path = os.path.join(f"./checkpoints/{args.taskName}", f"best_delta_{args.taskName}")

    # log residual branch configuration
    live_logger.info(
        "[DELTA] residual branch: "
        f"arch={_resolve_residual_arch(args)} "
        f"mode={_resolve_delta_residual_mode(args)} "
        f"sign_mode={_resolve_delta_sign_mode(args)} "
        f"denom_floor={float(getattr(args, 'delta_relative_denom_floor', 1.0) if getattr(args, 'delta_relative_denom_floor', 1.0) is not None else 1.0):.6f} "
        f"ratio_clip={float(getattr(args, 'delta_relative_ratio_clip', 0.0) or 0.0):.6f}"
    )

    # cleaned residual supervision configuration
    live_logger.info(
        "[DELTA] cleaned residual supervision: "
        f"enable={int(getattr(args, 'cleaned_residual_enable', 1))} "
        f"smooth_alpha={float(getattr(args, 'cleaned_residual_smooth_alpha', 0.6) or 0.0):.3f} "
        f"structured_mix={float(getattr(args, 'cleaned_residual_structured_mix', 0.35) or 0.0):.3f}"
    )

    # log temporal text auxiliary input configuration
    live_logger.info(
        "[DELTA] temporal text auxiliary input: "
        f"enable={int(getattr(args, 'delta_temporal_text_enable', 0))} "
        f"source={str(getattr(args, 'delta_temporal_text_source', 'refined') or 'refined')} "
        f"dim={int(getattr(args, 'delta_temporal_text_dim', 8) or 8)} "
        f"max_len={int(getattr(args, 'delta_temporal_text_max_len', 96) or 96)} "
        f"per_step_topk={int(getattr(args, 'delta_temporal_text_per_step_topk', 3) or 3)} "
        f"fuse_lambda={float(getattr(args, 'delta_temporal_text_fuse_lambda', 0.5) or 0.0):.3f} "
        f"freeze_encoder={int(getattr(args, 'delta_temporal_text_freeze_encoder', 1))}"
    )

    def _save_residual_best(epoch_idx: int, metric_now: float | None):
        metric_value = None if metric_now is None else float(metric_now)
        shutil.rmtree(best_delta_path, ignore_errors=True)

        save_checkpoint(
            best_delta_path,
            tokenizer,
            delta_model,
            base_model_id=args.base_model,
            tokenizer_id=args.tokenizer or args.base_model,
            train_cfg=train_cfg,
            optimizer=optim_delta,
            scheduler=scheduler_delta,
            epoch=int(epoch_idx),
            global_step=global_step,
            extra_meta={
                "mu_global": float(global_zstats["mu_global"]),
                "sigma_global": float(global_zstats["sigma_global"]),
                "delta_residual_mode": _resolve_delta_residual_mode(args),
                "delta_relative_denom_floor": float(
                    getattr(args, "delta_relative_denom_floor", 1.0)
                    if getattr(args, "delta_relative_denom_floor", 1.0) is not None
                    else 1.0
                ),
                "delta_relative_ratio_clip": float(getattr(args, "delta_relative_ratio_clip", 0.0) or 0.0),
                "delta_sign_mode": _resolve_delta_sign_mode(args),
                "residual_arch": _resolve_residual_arch(args),
                "delta_sign_external_enable": int(external_sign_enabled),
                "delta_sign_external_tau": float(getattr(args, "delta_sign_external_tau", 1.0)),
                "delta_temporal_text_source": str(getattr(args, "delta_temporal_text_source", "refined") or "refined"),
                "delta_temporal_text_unfreeze_last_n": int(getattr(args, "delta_temporal_text_unfreeze_last_n", 0)),
                "delta_text_fuse_mode": str(getattr(args, "delta_text_fuse_mode", "gated_add")),
                "delta_multimodal_arch": str(getattr(args, "delta_multimodal_arch", "summary_gated") or "summary_gated"),
                "delta_multimodal_fuse_lambda": float(max(0.0, getattr(args, "delta_multimodal_fuse_lambda", 1.0))),
            },
        )
        if external_sign_enabled:
            ext_sign_path = os.path.join(best_delta_path, "external_signnet.pt")
            torch.save(
                {
                    "state_dict": external_signnet_model.state_dict(),
                    "variant": str(getattr(external_signnet_model, "model_variant", "mlp")),
                    "history_len": int(max(1, getattr(args, "history_len", 1))),
                    "horizon": int(max(1, getattr(args, "horizon", 1))),
                    "structured_dim": int(max(0, getattr(args, "delta_structured_feature_dim", 12))),
                    "text_summary_dim": int(max(0, getattr(external_signnet_model, "text_summary_dim", 0))),
                    "task_type": str(getattr(external_signnet_model, "task_type", "binary_sign")),
                    "hidden_size": int(max(32, getattr(args, "delta_sign_external_hidden", 256))),
                    "dropout": float(max(0.0, getattr(args, "delta_sign_external_dropout", 0.1))),
                    "multimodal_fuse_lambda": float(max(0.0, getattr(args, "delta_multimodal_fuse_lambda", 1.0))),
                    "tau": float(getattr(args, "delta_sign_external_tau", 1.0)),
                    "decision_bias": float(
                        external_signnet_model.decision_bias.detach().cpu().item()
                    ) if hasattr(external_signnet_model, "decision_bias") else 0.0,
                },
                ext_sign_path,
            )

        with open(os.path.join(f"./checkpoints/{args.taskName}", "residual_pair.json"), "w", encoding="utf-8") as f:
            json.dump(
                {
                    "best_base": os.path.basename(best_base_path),
                    # "best_delta": f"best_delta_{args.taskName}",
                    # "best_tpl_id": int(tpl_id_now),
                    # "best_policy_name": str(policy_name_now),
                    "best_delta_alpha": float(best_delta_alpha),
                    "select_metric": str(args.select_metric),
                    "best_metric": metric_value,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

    def _dump_val_residual_debug_csv():
        if not val_residual_debug_csv_path:
            return
        evaluate_metrics_residual(
            base_model=base_backbone,
            delta_model=delta_model,
            external_signnet=external_signnet_model,
            tokenizer=tokenizer,
            temporal_text_tokenizer=temporal_text_tokenizer,
            data_loader=val_loader,
            templates=templates,
            tpl_id=tpl_id,
            args=args,
            global_zstats=global_zstats,
            news_df=news_df,
            policy_name=policy_name,
            policy_kw=policy_kw,
            device=device,
            volatility_bin=volatility_bin_val,
            testing=False,
            true_pred_csv_path=None,
            news_dropout=False,
            api_adapter=news_api_adapter,
            residual_debug_csv_path=val_residual_debug_csv_path,
            residual_debug_split="val",
        )
        live_logger.info(f"[DELTA][VAL_DEBUG_CSV] updated path={val_residual_debug_csv_path}")

    for epoch in range(delta_epochs):
        pbar = tqdm(train_loader, desc=f"[DELTA] Epoch {epoch+1}/{delta_epochs}")
        hard_reflect_mode = str(getattr(args, "hard_reflection_mode", "off")).lower().strip()
        hard_reflect_buffer = []

        for bidx, batch in enumerate(pbar):
            with torch.no_grad():
                history_z, _, _ = _z_batch_tensors(batch, args, global_zstats=global_zstats)
                history_z = history_z.to(device)
                base_pred = base_backbone(history_z).to(torch.float32)

            # ======= build delta inputs (with news) ======
            delta_inputs = build_delta_batch_inputs(
                batch=batch,
                tokenizer=tokenizer,
                temporal_text_tokenizer=temporal_text_tokenizer,
                templates=templates,
                tpl_id=tpl_id,
                args=args,
                global_zstats=global_zstats,
                news_df=news_df,
                policy_name=policy_name,
                policy_kw=policy_kw,
                volatility_bin=volatility_bin,
                testing=False,
                force_no_news=False,
                news_dropout=True,
                api_adapter=news_api_adapter,
            )
            ts_p = delta_inputs["ts_patches"]
            ts_pm = delta_inputs["ts_patch_mask"]
            targets_z = delta_inputs["targets_z"]
            prompt_texts_d = delta_inputs["prompt_texts"]
            news_counts_d = delta_inputs["news_counts"]
            structured_feats_d = delta_inputs["structured_feats"]
            temporal_text_ids_d = delta_inputs.get("temporal_text_ids")
            temporal_text_attn_d = delta_inputs.get("temporal_text_attn")
            temporal_text_step_mask_d = delta_inputs.get("temporal_text_step_mask")
            sample_debug_records = list(delta_inputs.get("sample_debug_records") or [])
            for i_debug, rec in enumerate(sample_debug_records):
                if i_debug < len(news_counts_d):
                    rec["news_count"] = int(float(news_counts_d[i_debug].detach().cpu()))
            _maybe_log_batch_news_debug(
                live_logger=live_logger,
                args=args,
                split="train",
                epoch_idx=epoch,
                batch_idx=bidx,
                sample_debug_records=sample_debug_records,
            )

            ts_p = ts_p.to(device)
            ts_pm = ts_pm.to(device)
            targets_z = targets_z.to(device)
            news_counts_d = news_counts_d.to(device=device, dtype=torch.float32)
            structured_feats_d = structured_feats_d.to(device=device, dtype=torch.float32)
            if temporal_text_ids_d is not None:
                temporal_text_ids_d = temporal_text_ids_d.to(device=device, dtype=torch.long)
            if temporal_text_attn_d is not None:
                temporal_text_attn_d = temporal_text_attn_d.to(device=device, dtype=torch.long)
            if temporal_text_step_mask_d is not None:
                temporal_text_step_mask_d = temporal_text_step_mask_d.to(device=device, dtype=torch.long)
            has_news = (news_counts_d > 0).to(dtype=torch.float32)
            signnet_history_z = history_z.to(torch.float32) if external_sign_enabled else None
            # ======= ============== =======
            delta_model.train()

            raw_delta_targets = _build_delta_targets(
                targets_z=targets_z,
                base_pred=base_pred,
                mu_global=float(global_zstats["mu_global"]),
                sigma_global=float(global_zstats["sigma_global"]),
                args=args,
            )
            if relative_mode:
                delta_targets = raw_delta_targets
                relative_state_targets = _build_relative_state_targets(
                    raw_delta_targets,
                    args,
                    structured_feats=structured_feats_d,
                )
            else:
                delta_targets = _build_cleaned_residual_targets(
                    raw_residual=raw_delta_targets,
                    structured_feats=structured_feats_d,
                    args=args,
                )
                relative_state_targets = None

            out_delta = delta_model(
                ts_patches=ts_p,
                ts_patch_mask=ts_pm,
                history_z=history_z,
                base_pred_z=base_pred,
                targets=None,
                head_mode="delta",
                rel_targets=None,
                rel_lambda=0.0,
                structured_feats=structured_feats_d,
                temporal_text_ids=temporal_text_ids_d,
                temporal_text_attn=temporal_text_attn_d,
                temporal_text_step_mask=temporal_text_step_mask_d,
            )
            delta_pred_model_real = out_delta["pred"].to(torch.float32)
            magnitude_real = out_delta.get("magnitude", delta_pred_model_real.abs()).to(torch.float32)
            if external_sign_enabled:
                with torch.no_grad():
                    ctrl_logits_real, ctrl_score_real = _run_external_signnet(
                        signnet_model=external_signnet_model,
                        history_z=signnet_history_z,
                        base_pred_z=base_pred,
                        structured_feats=structured_feats_d,
                        text_summary=out_delta.get("text_summary"),
                        text_strength=out_delta.get("text_strength"),
                        tau=float(getattr(args, "delta_sign_external_tau", 1.0)),
                    )
                delta_pred_real = _compose_delta_with_external_sign(
                    magnitude=magnitude_real,
                    sign_soft=ctrl_score_real,
                    delta_clip=float(getattr(delta_model, "delta_clip", getattr(args, "delta_clip", 0.0))),
                    residual_mode=_resolve_delta_residual_mode(args),
                    relative_delta=delta_pred_model_real,
                )
                if relative_mode:
                    state_logits_real = ctrl_logits_real.to(torch.float32)
                    sign_logits_real = None
                    sign_soft_real = None
                else:
                    sign_logits_real = ctrl_logits_real.to(torch.float32)
                    sign_soft_real = ctrl_score_real.to(torch.float32)
                    state_logits_real = None
            else:
                delta_pred_real = delta_pred_model_real
                direct_signed_mode = _direct_signed_delta_mode(args, delta_model)
                if relative_mode and not direct_signed_mode:
                    state_logits_real = out_delta.get(
                        "state_logits",
                        torch.zeros(delta_pred_model_real.size(0), delta_pred_model_real.size(1), 3, device=delta_pred_model_real.device, dtype=torch.float32),
                    ).to(torch.float32)
                    sign_logits_real = None
                    sign_soft_real = None
                else:
                    sign_logits_real = out_delta.get(
                        "sign_logits",
                        out_delta.get("direction_logits", delta_pred_model_real),
                    ).to(torch.float32)
                    sign_soft_real = out_delta.get(
                        "sign_soft",
                        out_delta.get("direction_score", torch.tanh(sign_logits_real)),
                    ).to(torch.float32)
                    state_logits_real = None
            news_available_mask = out_delta.get(
                "news_available_mask",
                has_news.unsqueeze(1),
            ).to(torch.float32)
            usable_news = news_available_mask.reshape(news_available_mask.size(0), -1).max(dim=1).values.clamp(0.0, 1.0)

            pred_real_z = _fuse_base_and_delta(
                base_pred_z=base_pred,
                delta_pred=delta_pred_real,
                args=args,
                mu_global=float(global_zstats["mu_global"]),
                sigma_global=float(global_zstats["sigma_global"]),
            )
            targets_z_typed = targets_z.to(torch.float32)
            true_residual_z = delta_targets.to(torch.float32)
            abs_residual_target = true_residual_z.abs()
            direct_signed_mode = _direct_signed_delta_mode(args, delta_model)
            if relative_mode and not direct_signed_mode:
                valid_sign_mask = torch.ones_like(abs_residual_target, dtype=torch.float32)
                sign_target_loss = None
                sign_target_bin = None
            else:
                sign_target_loss, sign_target_bin = _build_direction_target_pack(true_residual_z, args)
                sign_mode = str(getattr(args, "sign_label_mode", "hard") or "hard").lower().strip()
                if sign_mode == "windowed":
                    valid_sign_mask = torch.ones_like(abs_residual_target, dtype=torch.float32)
                else:
                    valid_sign_mask = (abs_residual_target > float(getattr(args, "delta_sign_eps", 0.03))).to(torch.float32)

            residual_mode = str(getattr(args, "residual_loss", "mae")).lower()
            loss_final = _point_loss(pred_real_z, targets_z_typed, mode=residual_mode)
            loss_total = loss_final
            if direct_signed_mode:
                loss_signed = _point_loss(delta_pred_real, true_residual_z, mode=residual_mode)
                loss_total = loss_total + loss_signed
                loss_relative_mag = None
                state_ce = None
                direction_ce = None
                confidence_consistency = None
                if _resolve_residual_arch(args) == "unified":
                    loss_relative_mag = _point_loss(magnitude_real, abs_residual_target, mode=residual_mode)
                    direction_logits_real = out_delta.get("direction_logits", sign_logits_real).to(torch.float32)
                    direction_score_real = out_delta.get("direction_score", sign_soft_real).to(torch.float32)
                    confidence_real = out_delta.get("confidence", torch.ones_like(delta_pred_real)).to(torch.float32)
                    direction_ce = F.binary_cross_entropy_with_logits(direction_logits_real, sign_target_loss)
                    confidence_consistency = _build_consistency_loss(
                        direction_score=direction_score_real,
                        magnitude=magnitude_real,
                        confidence=confidence_real,
                        delta_target=true_residual_z,
                    )
                    loss_total = (
                        loss_total
                        + loss_relative_mag
                        + float(getattr(args, "unified_direction_loss_weight", 0.3)) * direction_ce
                        + float(getattr(args, "unified_confidence_loss_weight", 0.1)) * confidence_consistency
                    )
            elif relative_mode:
                loss_relative_mag = _point_loss(magnitude_real, abs_residual_target, mode=residual_mode)
                loss_total = loss_total + loss_relative_mag
                if not external_sign_enabled:
                    state_ce = F.cross_entropy(
                        state_logits_real.reshape(-1, state_logits_real.size(-1)),
                        relative_state_targets.reshape(-1),
                        reduction="mean",
                    )
                    loss_total = loss_total + state_ce
                else:
                    state_ce = None
                direction_ce = None
                confidence_consistency = None
            else:
                loss_relative_mag = None
                state_ce = None
                direction_ce = None
                confidence_consistency = None

            if hard_reflect_mode not in {"off", "none"} and len(prompt_texts_d) > 0:
                err_real_batch = torch.abs(pred_real_z - targets_z_typed).mean(dim=1).detach().cpu().numpy()
                if err_real_batch.size > 0:
                    hard_i = int(np.argmax(err_real_batch))
                    hard_reflect_buffer.append(
                        {
                            "epoch": int(epoch + 1),
                            "batch_idx": int(bidx),
                            "error_z_mae": float(err_real_batch[hard_i]),
                            "prompt": str(prompt_texts_d[hard_i]),
                            "target_time": str(batch["target_time"][hard_i]),
                            "has_news": int(float(has_news[hard_i].detach().cpu()) > 0.5),
                        }
                    )

            loss = loss_total / args.grad_accum
            loss.backward()

            loss_window.append(float(loss.detach().cpu()))
            if global_step % 10 == 0:
                avg_train_loss = sum(loss_window) / len(loss_window)
                sign_acc = float(
                    (
                        _masked_multiclass_accuracy_from_logits(state_logits_real, relative_state_targets, valid_sign_mask)
                        if relative_mode and not direct_signed_mode
                        else _masked_binary_accuracy_from_logits(sign_logits_real, sign_target_bin, valid_sign_mask)
                    )
                    .detach()
                    .cpu()
                )
                mag_mean = float(magnitude_real.mean().detach().cpu())
                mag_true_mean = float(abs_residual_target.mean().detach().cpu())
                delta_abs_mean = float(delta_pred_real.abs().mean().detach().cpu())
                news_frac = float(usable_news.mean().detach().cpu())
                postfix = {
                    "train_loss": f"{avg_train_loss:.6f}",
                    "final": float(loss_final.detach().cpu()),
                    "mag": mag_mean,
                    "true": mag_true_mean,
                    "delta": delta_abs_mean,
                    "news": news_frac,
                }
                if relative_mode and not direct_signed_mode:
                    postfix["state_acc"] = sign_acc
                else:
                    postfix["sgn"] = sign_acc
                pbar.set_postfix(**postfix)

            if (global_step + 1) % args.grad_accum == 0:
                grad_clip = float(getattr(args, "delta_grad_clip", 0.0))
                if grad_clip > 0:
                    clip_grad_norm_((p for p in delta_model.parameters() if p.requires_grad), grad_clip)
                optim_delta.step()
                if args.scheduler == 1:
                    scheduler_delta.step()
                optim_delta.zero_grad(set_to_none=True)

            global_step += 1

        if delta_val_mode == "each_epoch":
            # end-of-epoch eval (combined)
            val_loss, val_mse, val_mae, base_val_loss, base_val_mse, base_val_mae = evaluate_metrics_residual(
                base_model=base_backbone,
                delta_model=delta_model,
                external_signnet=external_signnet_model,
                tokenizer=tokenizer,
                temporal_text_tokenizer=temporal_text_tokenizer,
                data_loader=val_loader,
                templates=templates,
                tpl_id=tpl_id,
                args=args,
                global_zstats=global_zstats,
                news_df=news_df,
                policy_name=policy_name,
                policy_kw=policy_kw,
                device=device,
                volatility_bin=volatility_bin_val,
                testing=False,
                true_pred_csv_path=None,
                news_dropout=False,
                api_adapter=news_api_adapter,
            )
            val_loss_per_epoch.append(val_loss)
            mse_loss_per_epoch.append(val_mse)
            mae_loss_per_epoch.append(val_mae)

            live_logger.info(
                f"[DELTA][EVAL] epoch={epoch+1} tpl_id={tpl_id} policy={policy_name} "
                f"val_loss(zMSE)={val_loss:.6f} "
                f"val_mse(raw)={val_mse:.6f} val_mae(raw)={val_mae:.6f}"
            )
            _log_last_residual_eval_diag(args, live_logger, f"[DELTA][EVAL_DIAG][VAL] epoch={epoch+1}")
            live_logger.info(
                f"[DELTA][BASE_ONLY][VAL] epoch={epoch+1} "
                f"loss(zMAE)={base_val_loss:.6f} mse(raw)={base_val_mse:.6f} mae(raw)={base_val_mae:.6f}"
            )
            metric_now = _select_metric(val_loss, val_mse, val_mae, args.select_metric)
            delta_vs_base = float(metric_now - base_ref_metric)
            live_logger.info(
                f"[DELTA][COMPARE][VAL] epoch={epoch+1} metric={metric_now:.6f} "
                f"delta_vs_base={delta_vs_base:+.6f}"
            )

            should_save = (not has_saved_delta) or (metric_now < best_metric - 1e-6)
            if should_save:
                best_metric = float(metric_now)
                stale_rounds = 0
                has_saved_delta = True
                best_tpl_id = tpl_id
                best_policy_name = policy_name
                best_delta_alpha = 1.0
                _save_residual_best(
                    epoch_idx=epoch,
                    metric_now=best_metric,
                )
                _dump_val_residual_debug_csv()
                live_logger.info(
                    f"[DELTA] New best delta saved: {best_delta_path} "
                    f"({args.select_metric}={best_metric:.6f}, "
                    f"delta_vs_base={best_metric - base_ref_metric:+.6f})"
                )
            else:
                stale_rounds += 1
                if early_stop_patience > 0:
                    live_logger.info(
                        f"[DELTA] stale_rounds={stale_rounds}/{early_stop_patience} "
                        f"best={best_metric:.6f} delta_vs_base={best_metric - base_ref_metric:+.6f}"
                    )

            if early_stop_patience > 0 and stale_rounds >= early_stop_patience:
                live_logger.info(f"[DELTA] Early stopping triggered at epoch {epoch+1}.")
                break

        if hard_reflect_mode not in {"off", "none"} and len(hard_reflect_buffer) > 0:
            top_k = int(max(1, getattr(args, "hard_reflection_topk", 8)))
            hard_reflect_buffer = sorted(
                hard_reflect_buffer,
                key=lambda x: float(x.get("error_z_mae", 0.0)),
                reverse=True,
            )[:top_k]
            reflections = reflect_hard_samples(
                hard_samples=hard_reflect_buffer,
                mode=hard_reflect_mode,
                api_adapter=news_api_adapter,
            )
            live_logger.info(
                f"[DELTA][REFLECT] epoch={epoch+1} mode={hard_reflect_mode} "
                f"hard_samples={len(hard_reflect_buffer)} reflections={len(reflections)}"
            )

        _save_refine_cache(args, live_logger=live_logger, force=False)
        _save_structured_cache(args, live_logger=live_logger, force=False)

        if epoch == 0:
            _log_prompt_stats_if_available(
                live_logger,
                dataStatistic,
                "---------------------trainset and valset prompt statistics--------------------------------",
                "[DELTA][PROMPT_STATS] skipped: DELTA prompt path is disabled in this stage.",
            )

    if not has_saved_delta:
        final_epoch = max(0, int(delta_epochs) - 1)
        if delta_val_mode == "end_only":
            val_loss, val_mse, val_mae, base_val_loss, base_val_mse, base_val_mae = evaluate_metrics_residual(
                base_model=base_backbone,
                delta_model=delta_model,
                external_signnet=external_signnet_model,
                tokenizer=tokenizer,
                temporal_text_tokenizer=temporal_text_tokenizer,
                data_loader=val_loader,
                templates=templates,
                tpl_id=tpl_id,
                args=args,
                global_zstats=global_zstats,
                news_df=news_df,
                policy_name=policy_name,
                policy_kw=policy_kw,
                device=device,
                volatility_bin=volatility_bin_val,
                testing=False,
                true_pred_csv_path=None,
                news_dropout=False,
                api_adapter=news_api_adapter,
            )
            metric_now = _select_metric(val_loss, val_mse, val_mae, args.select_metric)
            best_metric = float(metric_now)
            has_saved_delta = True
            best_tpl_id = tpl_id
            best_policy_name = policy_name
            _save_residual_best(
                epoch_idx=final_epoch,
                metric_now=best_metric,
            )
            _dump_val_residual_debug_csv()
            live_logger.info(
                f"[DELTA][VAL] end_only: val_loss(zMSE)={val_loss:.6f} "
                f"val_mse(raw)={val_mse:.6f} val_mae(raw)={val_mae:.6f}"
            )
            _log_last_residual_eval_diag(args, live_logger, "[DELTA][EVAL_DIAG][VAL] end_only")
            live_logger.info(
                f"[DELTA][BASE_ONLY][VAL] end_only: loss(zMAE)={base_val_loss:.6f} "
                f"mse(raw)={base_val_mse:.6f} mae(raw)={base_val_mae:.6f}"
            )
            live_logger.info(
                f"[DELTA] Saved end_only best delta: {best_delta_path} "
                f"({args.select_metric}={best_metric:.6f})"
            )
        elif delta_val_mode == "none":
            has_saved_delta = True
            best_tpl_id = tpl_id
            best_policy_name = policy_name
            best_metric = float("nan")
            _save_residual_best(
                epoch_idx=final_epoch,
                metric_now=None,
            )
            live_logger.info(
                f"[DELTA][VAL] skipped (mode={delta_val_mode}); "
                f"saved final checkpoint: {best_delta_path} (epoch={final_epoch+1})"
            )
        else:
            # safety fallback for unexpected state in each_epoch mode
            val_loss, val_mse, val_mae, base_val_loss, base_val_mse, base_val_mae = evaluate_metrics_residual(
                base_model=base_backbone,
                delta_model=delta_model,
                external_signnet=external_signnet_model,
                tokenizer=tokenizer,
                temporal_text_tokenizer=temporal_text_tokenizer,
                data_loader=val_loader,
                templates=templates,
                tpl_id=tpl_id,
                args=args,
                global_zstats=global_zstats,
                news_df=news_df,
                policy_name=policy_name,
                policy_kw=policy_kw,
                device=device,
                volatility_bin=volatility_bin_val,
                testing=False,
                true_pred_csv_path=None,
                news_dropout=False,
                api_adapter=news_api_adapter,
            )
            metric_now = _select_metric(val_loss, val_mse, val_mae, args.select_metric)
            best_metric = float(metric_now)
            has_saved_delta = True
            best_tpl_id = tpl_id
            best_policy_name = policy_name
            _save_residual_best(
                epoch_idx=final_epoch,
                metric_now=best_metric,
            )
            _dump_val_residual_debug_csv()
            live_logger.info(
                f"[DELTA][VAL] fallback eval: val_loss(zMSE)={val_loss:.6f} "
                f"val_mse(raw)={val_mse:.6f} val_mae(raw)={val_mae:.6f}"
            )
            _log_last_residual_eval_diag(args, live_logger, "[DELTA][EVAL_DIAG][VAL] fallback")
            live_logger.info(
                f"[DELTA][BASE_ONLY][VAL] fallback: loss(zMAE)={base_val_loss:.6f} "
                f"mse(raw)={base_val_mse:.6f} mae(raw)={base_val_mae:.6f}"
            )

    dataStatistic.clear()
    _save_refine_cache(args, live_logger=live_logger, force=True)
    _save_structured_cache(args, live_logger=live_logger, force=True)

    # TEST (combined with best delta; base computed via adapter_off)
    if test_loader is not None:
        del delta_model
         # ---- free training-time GPU objects ----
        # del delta_model
        # del optim_delta
        # if scheduler_delta is not None:
        #     del scheduler_delta

        gc.collect()
        torch.cuda.empty_cache()

        best_delta_path = os.path.join(f"./checkpoints/{args.taskName}", f"best_delta_{args.taskName}")
        if not os.path.exists(best_delta_path):
            live_logger.info("[DELTA] best delta checkpoint not found; fallback to base-only test.")
            test_loss, test_mse, test_mae = evaluate_metrics_backbone(
                base_backbone=base_backbone,
                data_loader=test_loader,
                args=args,
                global_zstats=global_zstats,
                device=device,
                testing=True,
                true_pred_csv_path=true_pred_csv_path,
                filename=bundle["test_filename"],
            )
            tqdm.write(f"[TEST][FALLBACK-BASE] loss(zMSE)={test_loss:.6f} mse(raw)={test_mse:.6f} mae(raw)={test_mae:.6f}")
            live_logger.info(
                f"[TEST][FALLBACK-BASE] loss(zMSE)={test_loss:.6f} mse(raw)={test_mse:.6f} mae(raw)={test_mae:.6f}"
            )
            record_test_results_csv(args, live_logger, test_mse, test_mae, base_mse=test_mse, base_mae=test_mae)
            draw_pred_true(live_logger, args, true_pred_csv_path)
            _save_refine_cache(args, live_logger=live_logger, force=True)
            _save_structured_cache(args, live_logger=live_logger, force=True)
            return {
                "test_loader": test_loader,
                "device": device,
                "live_logger": live_logger,
                "best_tpl_id": best_tpl_id,
                "best_policy_name": best_policy_name,
                "tpl_id": tpl_id,
                "policy_name": policy_name,
                "templates": templates,
                "news_df": news_df,
                "policy_kw": policy_kw,
                "volatility_bin_test": volatility_bin_test,
                "true_pred_csv_path": true_pred_csv_path,
                "global_zstats": global_zstats,
            }

        tok_d, temporal_tok_d, model_best = load_checkpoint(
            best_delta_path,
            _single_device_map(args),
            False,
            head_mlp=args.head_mlp,
            hd=args.head_dropout,
            pd=args.patch_dropout


        )

        tokenizer = tok_d
        temporal_text_tokenizer = temporal_tok_d
        model_best.to(device)
        model_best.eval()

        delta_meta = {}
        delta_meta_path = os.path.join(best_delta_path, "meta.json")
        if os.path.isfile(delta_meta_path):
            try:
                with open(delta_meta_path, "r", encoding="utf-8") as f:
                    delta_meta = json.load(f)
            except Exception:
                delta_meta = {}
        global_zstats_eval = _coerce_global_zstats(delta_meta, args, required=False)
        if global_zstats_eval is None:
            global_zstats_eval = global_zstats
        else:
            live_logger.info(
                "[DELTA] loaded global z-score stats from delta checkpoint meta: "
                f"mu_global={global_zstats_eval['mu_global']:.6f}, "
                f"sigma_global={global_zstats_eval['sigma_global']:.6f}"
            )

        live_logger.info(
            f"Loaded best DELTA model for testing (final = {base_meta.get('backbone_name')} + delta(adapter_on,news))."
        )

        tpl_for_test = tpl_id
        pol_for_test = policy_name
        (
            test_loss,
            test_mse,
            test_mae,
            base_test_loss,
            base_test_mse,
            base_test_mae,
        ) = evaluate_metrics_residual(
            base_model=base_backbone,
            delta_model=model_best,
            external_signnet=external_signnet_model,
            tokenizer=tokenizer,
            temporal_text_tokenizer=temporal_text_tokenizer,
            data_loader=test_loader,
            templates=templates,
            tpl_id=tpl_for_test,
            args=args,
            global_zstats=global_zstats_eval,
            news_df=news_df,
            policy_name=pol_for_test,
            policy_kw=policy_kw,
            device=device,
            volatility_bin=volatility_bin_test,
            testing=True,
            true_pred_csv_path=true_pred_csv_path,
            news_dropout=False,
            filename=bundle["test_filename"],
            api_adapter=news_api_adapter,
            residual_debug_csv_path=test_residual_debug_csv_path,
            residual_debug_split="test",
        )

        _log_prompt_stats_if_available(
            live_logger,
            dataStatistic,
            "---------------------testset prompt statistics--------------------------------",
            "[DELTA][PROMPT_STATS] skipped: DELTA test path did not build model-consumed prompts.",
        )

        tqdm.write(
            f"[TEST][FINAL] loss(zMSE)={test_loss:.6f} mse(raw)={test_mse:.6f} mae(raw)={test_mae:.6f}"
        )
        live_logger.info(
            f"[TEST][FINAL] loss(zMSE)={test_loss:.6f} mse(raw)={test_mse:.6f} mae(raw)={test_mae:.6f}"
        )
        _log_last_residual_eval_diag(args, live_logger, "[TEST][FINAL_DIAG]")
        live_logger.info(
            f"[TEST][BASE_ONLY] loss(zMAE)={base_test_loss:.6f} mse(raw)={base_test_mse:.6f} "
            f"mae(raw)={base_test_mae:.6f}"
        )
        if test_residual_debug_csv_path:
            live_logger.info(f"[DELTA][TEST_DEBUG_CSV] updated path={test_residual_debug_csv_path}")

        record_test_results_csv(args, live_logger, test_mse, test_mae, base_mse=base_test_mse, base_mae=base_test_mae)
        draw_pred_true(live_logger, args, true_pred_csv_path)
        _save_refine_cache(args, live_logger=live_logger, force=True)
        _save_structured_cache(args, live_logger=live_logger, force=True)

    return {
        "test_loader":test_loader,
        "device": device,
        "live_logger": live_logger,
        "best_tpl_id":best_tpl_id,
        "best_policy_name":best_policy_name,
        "tpl_id":tpl_id,
        "policy_name":policy_name,
        "templates":templates,
        "news_df":news_df,
        "policy_kw":policy_kw,
        "volatility_bin_test": volatility_bin_test,
        "true_pred_csv_path": true_pred_csv_path,
        "global_zstats": global_zstats,
    }
