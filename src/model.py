# model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, PeftModel
import os
import json
END_TOKEN = "<END>"

    

class TSForecastRegressor(nn.Module):
    """
    Text + (time-series patches as soft tokens) -> LLM -> pool patch hidden states -> regression head -> H values

    - input_ids / attention_mask: tokenized prompt (instruction + news + any brief history text)
    - ts_patches: float tensor (B, P, patch_dim)  (z-scored patches)
    - ts_patch_mask: 0/1 mask (B, P) for padded patches
    - targets: float tensor (B, H) (z-scored targets). If provided, returns MSE loss in z-space.
    """

    def __init__(
        self,
        lm,
        horizon: int,
        patch_dim: int,
        hidden_size: int,
        patch_dropout: float,
        head_dropout: float,
        head_mlp: bool = False,
    ):
        super().__init__()
        self.lm = lm
        self.horizon = int(horizon)
        self.patch_dim = int(patch_dim)
        self.hidden_size = int(hidden_size)

        self.patch_proj = nn.Linear(self.patch_dim, self.hidden_size)
        self.patch_drop = nn.Dropout(float(patch_dropout))

        self.rel_head = nn.Linear(self.hidden_size, 1)
        

        # -------------- base head ----------------------------
        if head_mlp:
            print("using mlp")
            self.head = nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.GELU(),
                nn.Dropout(float(head_dropout)),
                nn.Linear(self.hidden_size, self.horizon),
            )
        else:
            print("not using mlp")
            self.head_drop = nn.Dropout(float(head_dropout))
            self.head = nn.Linear(self.hidden_size, self.horizon)
        self.base_head = self.head
        # ----------------- residual head ------------------------
        # build delta_head with same architecture as base head
        if head_mlp:
            self.delta_head = nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.GELU(),
                nn.Dropout(float(head_dropout)),
                nn.Linear(self.hidden_size, self.horizon),
            )
            # zero-init last layer for residual learning
            nn.init.zeros_(self.delta_head[-1].weight)
            nn.init.zeros_(self.delta_head[-1].bias)
        else:
            self.delta_head_drop = nn.Dropout(float(head_dropout))
            self.delta_head = nn.Linear(self.hidden_size, self.horizon)
            nn.init.zeros_(self.delta_head.weight)
            nn.init.zeros_(self.delta_head.bias)

        # dtype/device align
        lm_dtype = next(self.lm.parameters()).dtype
        self.base_head = self.base_head.to(dtype=lm_dtype)
        self.delta_head = self.delta_head.to(dtype=lm_dtype)
        self.rel_head = self.rel_head.to(dtype=lm_dtype)
        if not head_mlp:
            self.delta_head_drop = self.delta_head_drop  # keep

    def _pool_patch_hidden(self, last_hidden: torch.Tensor, ts_patch_mask: torch.Tensor) -> torch.Tensor:
        """
        last_hidden: (B, T_text + P, H)
        ts_patch_mask: (B, P) 0/1
        returns pooled: (B, H)
        """
        B, seq_len, H = last_hidden.shape
        P = ts_patch_mask.size(1)
        patch_hid = last_hidden[:, -P:, :]  # (B, P, H)

        m = ts_patch_mask.to(dtype=patch_hid.dtype).unsqueeze(-1)  # (B, P, 1)
        denom = m.sum(dim=1).clamp_min(1.0)  # (B, 1)

        pooled = (patch_hid * m).sum(dim=1) / denom  # (B, H)
        return pooled

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        ts_patches: torch.Tensor,
        ts_patch_mask: torch.Tensor,
        targets: torch.Tensor | None = None,
        head_mode: str = "base",   # NEW
        rel_targets: torch.Tensor | None = None,
        rel_lambda: float = 0.0,
    ):
        # text token embeddings (dtype usually bf16)
        tok_emb = self.lm.get_input_embeddings()(input_ids)  # (B, T, H)
        tok_dtype = tok_emb.dtype

        # ---关键：让 patch 分支 dtype 对齐---
        proj_dtype = self.patch_proj.weight.dtype
        if ts_patches.dtype != proj_dtype:
            ts_patches = ts_patches.to(dtype=proj_dtype)

        patch_emb = self.patch_proj(ts_patches)  # (B, P, H)  dtype = proj_dtype
        patch_emb = self.patch_drop(patch_emb)

        # 再对齐到 tok_emb 的 dtype，保证 cat 后 inputs_embeds 统一 dtype
        if patch_emb.dtype != tok_dtype:
            patch_emb = patch_emb.to(dtype=tok_dtype)

        if self.training and getattr(self, "patch_mask_p", 0.0) > 0:
            # (B,P,1)
            keep = (torch.rand(patch_emb.size(0), patch_emb.size(1), 1, device=patch_emb.device) > self.patch_mask_p).to(patch_emb.dtype)
            patch_emb = patch_emb * keep
        # concat as "soft tokens": [text_tokens, patch_tokens]
        inputs_embeds = torch.cat([tok_emb, patch_emb], dim=1)  # (B, T+P, H)
        attn = torch.cat([attention_mask, ts_patch_mask], dim=1)  # (B, T+P)

        outputs = self.lm(
            inputs_embeds=inputs_embeds,
            attention_mask=attn,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )

        last_hidden = outputs.hidden_states[-1]  # (B, T+P, H)
        pooled = self._pool_patch_hidden(last_hidden, ts_patch_mask)  # (B, H)
        

        if head_mode == "base":
            if isinstance(self.base_head, nn.Linear):
                pred = self.base_head(self.head_drop(pooled))
            else:
                pred = self.base_head(pooled)
        elif head_mode == "delta":
            if isinstance(self.delta_head, nn.Linear):
                pred = self.delta_head(self.delta_head_drop(pooled))
            else:
                pred = self.delta_head(pooled)
        else:
            raise ValueError(f"Unknown head_mode={head_mode}")
        
    
        rel_logit = self.rel_head(pooled).squeeze(-1)   # (B,)
        out = {"pred": pred, "rel_logits": rel_logit}

        loss = None
        if targets is not None:
            if targets.dtype != pred.dtype:
                targets = targets.to(dtype=pred.dtype)
            loss_fore = F.l1_loss(pred, targets, reduction="mean")
            loss = loss_fore

        if rel_targets is not None:
            rel_targets = rel_targets.to(device=rel_logit.device, dtype=rel_logit.dtype)
            loss_rel = F.binary_cross_entropy_with_logits(rel_logit, rel_targets)
            loss = rel_lambda * loss_rel if loss is None else (loss + rel_lambda * loss_rel)

        if loss is not None:
            out["loss"] = loss
            out["loss_fore"] = loss_fore if targets is not None else None
            out["loss_rel"] = loss_rel if rel_targets is not None else None

        return out


def load_llama_lora(
    base_model: str,
    tokenizer_id: str,
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
    target_modules,
    load_in_4bit: bool = False,
    gradient_checkpointing: bool = False,
    max_seq_len: int = 1536,
    device=None,
    horizon: int = 48,
    patch_dim: int = 4,
    patch_dropout: float = 0.0,
    head_dropout: float = 0.0,
    head_mlp: bool = False,
):
    tok = AutoTokenizer.from_pretrained(tokenizer_id or base_model, use_fast=True)

    # pad token
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
        tok.pad_token_id = tok.eos_token_id

    # (Optional) keep END token for backward-compat; not used by regressor
    special = tok.special_tokens_map.get("additional_special_tokens", [])
    if END_TOKEN not in special:
        tok.add_special_tokens({"additional_special_tokens": [END_TOKEN]})

    base = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if device is None else None,
        load_in_4bit=load_in_4bit,
    )

    # IMPORTANT: resize embeddings after adding tokens
    base.resize_token_embeddings(len(tok))

    if gradient_checkpointing:
        base.gradient_checkpointing_enable()

    peft_cfg = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )
    lm = get_peft_model(base, peft_cfg)
    lm.print_trainable_parameters()

    hidden_size = lm.config.hidden_size

    model = TSForecastRegressor(
        lm=lm,
        horizon=int(horizon),
        patch_dim=int(patch_dim),
        hidden_size=int(hidden_size),
        patch_dropout=float(patch_dropout),
        head_dropout=float(head_dropout),
        head_mlp=bool(head_mlp),
    )

    return tok, model



def save_checkpoint(
    ckpt_dir: str,
    tok,
    model,                      # TSForecastRegressor
    base_model_id: str,
    tokenizer_id: str | None,
    lora_cfg: dict,
    optimizer=None,
    scheduler=None,
    epoch: int | None = None,
    global_step: int | None = None,
):
    os.makedirs(ckpt_dir, exist_ok=True)

    # 1) tokenizer（包含你 add_special_tokens 之后的版本）
    tok_dir = os.path.join(ckpt_dir, "tokenizer")
    tok.save_pretrained(tok_dir)

    # 2) LoRA adapter（只保存 PEFT 参数，体积小）——HF 推荐用 save_pretrained() 复用 :contentReference[oaicite:0]{index=0}
    adapter_dir = os.path.join(ckpt_dir, "adapter")
    model.lm.save_pretrained(adapter_dir, safe_serialization=True)

    # 3) 你自定义的回归部分（patch_proj + head）
    reg_path = os.path.join(ckpt_dir, "regressor.pt")
    torch.save(
        {
            "patch_proj": model.patch_proj.state_dict(),
            "head": model.base_head.state_dict(),           # base
            "delta_head": model.delta_head.state_dict(),    # NEW
            "rel_head": model.rel_head.state_dict(),
            "horizon": model.horizon,
            "patch_dim": model.patch_dim,
            "hidden_size": model.hidden_size,
        },
        reg_path,
    )

    # 4) meta（用于“精确复现”加载）
    meta = {
        "base_model_id": base_model_id,
        "tokenizer_id": tokenizer_id or base_model_id,
        "lora_cfg": lora_cfg,  # r/alpha/dropout/target_modules 等
        "horizon": int(model.horizon),
        "patch_dim": int(model.patch_dim),
        "hidden_size": int(model.hidden_size),
    }
    with open(os.path.join(ckpt_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    # 5)（可选）断点续训：optimizer/scheduler/epoch/step/RNG
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


def load_checkpoint(
    ckpt_dir: str,
    load_in_4bit: bool = False,
    gradient_checkpointing: bool = False,
    device_map=None,
    is_trainable: bool = False,   # True=继续训练；False=纯推理
    head_mlp = False,
    hd: float = 0.0,
    pd: float = 0.0
):
    # 0) meta
    with open(os.path.join(ckpt_dir, "meta.json"), "r", encoding="utf-8") as f:
        meta = json.load(f)

    base_model_id = meta["base_model_id"]
    horizon = int(meta["horizon"])
    patch_dim = int(meta["patch_dim"])

    # 1) tokenizer（加载保存过的）
    tok = AutoTokenizer.from_pretrained(os.path.join(ckpt_dir, "tokenizer"), use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
        tok.pad_token_id = tok.eos_token_id

    # 2) base model（注意 dtype / device_map / 量化要和你训练环境匹配）
    base = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if device_map is None else device_map,
        load_in_4bit=load_in_4bit,
    )

    # 关键：对齐 embedding size（你训练时 add_special_tokens 后 resize 过）
    base.resize_token_embeddings(len(tok))

    if gradient_checkpointing:
        base.gradient_checkpointing_enable()

    # 3) attach LoRA adapter（从 adapter/ 恢复）:contentReference[oaicite:2]{index=2}
    adapter_dir = os.path.join(ckpt_dir, "adapter")
    lm = PeftModel.from_pretrained(base, adapter_dir, is_trainable=is_trainable)

    # 4) rebuild TSForecastRegressor 并加载 regressor 权重
    hidden_size = lm.config.hidden_size
    model = TSForecastRegressor(
        lm=lm,
        horizon=horizon,
        patch_dim=patch_dim,
        hidden_size=hidden_size,
        patch_dropout=pd,
        head_dropout=hd,
        head_mlp=head_mlp,
    )

    reg = torch.load(os.path.join(ckpt_dir, "regressor.pt"), map_location="cpu")
    model.patch_proj.load_state_dict(reg["patch_proj"], strict=True)
    model.head.load_state_dict(reg["head"], strict=True)
    if "delta_head" in reg:
        model.delta_head.load_state_dict(reg["delta_head"], strict=True)
    else:
        # init as zero residual if checkpoint doesn't have it (base checkpoints)
        for p in model.delta_head.parameters():
            nn.init.zeros_(p)

    model.rel_head.load_state_dict(reg["rel_head"], strict=True)
    # ✅ 关键：移动自定义层到GPU
    
    emb = model.lm.get_input_embeddings().weight
    model.patch_proj = model.patch_proj.to(emb.device, dtype=emb.dtype)
    model.head = model.head.to(emb.device, dtype=emb.dtype)
    model.delta_head = model.delta_head.to(emb.device, dtype=emb.dtype)
    model.rel_head = model.rel_head.to(emb.device, dtype=emb.dtype)



    return tok, model

# 6) （可选）加载训练状态
def load_trainer_state(ckpt_dir: str, optimizer, scheduler=None):
    path = os.path.join(ckpt_dir, "trainer_state.pt")
    if not os.path.exists(path):
        return None

    st = torch.load(path, map_location="cpu")
    optimizer.load_state_dict(st["optimizer"])
    if scheduler is not None and "scheduler" in st:
        scheduler.load_state_dict(st["scheduler"])

    torch.set_rng_state(st["rng_state"])
    if torch.cuda.is_available() and "cuda_rng_state_all" in st:
        torch.cuda.set_rng_state_all(st["cuda_rng_state_all"])

    return st  # 里面有 epoch/global_step