# run.py
import argparse
# from src.trainer import main as main_train
from src.base_delta_decoouple_trainer import main as main_train
# from src.original_residual_trainer import main as main_train

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Cross-domain LLaMA Forecasting with RL')

    
    parser.add_argument('--rel_lambda', type=float, default=0.3, help='weight for relative loss')
    parser.add_argument('--rel_supervise_lambda', type=float, default=0.5,
                        help='weight for supervised rel_head BCE in delta stage')

    parser.add_argument('--delta_null_lambda', type=float, default=0.0001, help='weight for ||delta_pred(no-news)|| shrink')
    parser.add_argument('--delta_margin_lambda', type=float, default=1.0, help='weight for counterfactual margin loss')
    parser.add_argument('--delta_adv_margin', type=float, default=0.02, help='margin in z-space: err_null >= err_real + margin')
    parser.add_argument('--delta_non_degrade_lambda', type=float, default=1.0, help='weight for non-degradation guard vs base')
    parser.add_argument('--delta_non_degrade_margin', type=float, default=0.0, help='margin in z-space: err_real <= err_base - margin')

    parser.add_argument('--news_gate_enable', type=int, default=1, choices=[0, 1], help='enable sample-wise news gate')
    parser.add_argument('--news_gate_temperature', type=float, default=1.0, help='temperature for sigmoid news gate')
    parser.add_argument('--news_gate_floor', type=float, default=0.0, help='lower bound for gate value to avoid over-shrink')
    parser.add_argument('--gate_lambda', type=float, default=0.2, help='weight for gate pseudo-label BCE loss')
    parser.add_argument('--gate_null_lambda', type=float, default=0.1, help='weight forcing no-news gate toward zero')
    parser.add_argument('--delta_gate_init_bias', type=float, default=0.0, help='init bias for horizon-wise delta gate head')
    parser.add_argument('--delta_internal_gate', type=int, default=1, choices=[0, 1],
                        help='enable internal delta gating in model head (1=on, 0=bypass gate/rel/clip)')
    parser.add_argument('--delta_head_init_std', type=float, default=0.01, help='std for delta head weight init')
    parser.add_argument('--delta_clip', type=float, default=3.0, help='tanh clip for delta outputs in z-space (<=0 to disable)')
    parser.add_argument('--delta_news_tail_tokens', type=int, default=160, help='how many tail text tokens to pool as news context')
    parser.add_argument('--delta_rel_floor', type=float, default=0.05, help='minimum multiplicative factor from relevance gate')

    parser.add_argument('--cf_pseudo_margin', type=float, default=0.01, help='counterfactual gain margin for pseudo labels')
    parser.add_argument('--cf_pseudo_temp', type=float, default=0.2, help='temperature for soft pseudo labels')
    parser.add_argument('--cf_pseudo_hard', type=int, default=0, choices=[0, 1], help='use hard(1)/soft(0) pseudo labels')
    parser.add_argument('--cf_min_weight', type=float, default=0.30, help='minimum residual weight for samples with news')
    parser.add_argument('--delta_cold_start_steps', type=int, default=200, help='force sample_w=1 for first N delta steps')
    parser.add_argument('--delta_null_warmup_steps', type=int, default=500, help='delay null loss for first N delta steps')
    parser.add_argument('--delta_null_ramp_steps', type=int, default=500, help='ramp null loss weight for next N delta steps')
    parser.add_argument('--news_contrastive_lambda', type=float, default=0.1, help='weight for news-vs-null rel-logit contrastive loss')
    parser.add_argument('--base_pred_noise', type=float, default=0.5, help='std of Gaussian noise added to base_pred in delta training')
    parser.add_argument('--delta_warmup_epochs', type=int, default=2,
                        help='disable gate/counterfactual regularizers in first N delta epochs')
    parser.add_argument('--delta_curriculum_epochs', type=int, default=3, help='ramp-up epochs for delta constraints')
    parser.add_argument('--delta_grad_clip', type=float, default=1.0, help='grad clip norm for delta stage (<=0 to disable)')
    parser.add_argument('--delta_violation_cap', type=float, default=1.0, help='cap per-sample margin/non-degrade hinge value (>0 to enable)')
    parser.add_argument('--delta_lora_lr_scale', type=float, default=0.3, help='lr scale for LoRA params in delta stage')
    parser.add_argument('--delta_head_lr_scale', type=float, default=1.0, help='lr scale for delta/rel heads in delta stage')
    parser.add_argument('--delta_other_lr_scale', type=float, default=0.5, help='lr scale for other trainable params in delta stage')
    parser.add_argument('--delta_freeze_feature_modules', type=int, default=0, choices=[0, 1],
                        help='freeze patch/pooling feature modules in delta stage (legacy behavior)')
    parser.add_argument(
        '--delta_mode',
        type=str,
        default='regression',
        choices=['regression', 'kernel_tokens'],
        help='delta prediction mode: regression head (legacy) or kernel parameter tokens',
    )
    parser.add_argument(
        '--delta_fusion_mode',
        type=str,
        default='add',
        choices=['add', 'mul_z', 'mul_raw'],
        help='how to fuse base and delta in kernel eval/inference: add, multiplicative on z-space, or multiplicative on raw-space',
    )
    parser.add_argument(
        '--delta_mul_scale',
        type=float,
        default=1.0,
        help='scale on token-derived delta before multiplicative fusion (coeff = 1 + scale*delta)',
    )
    parser.add_argument(
        '--delta_mul_coeff_min',
        type=float,
        default=0.05,
        help='lower bound for multiplicative coefficient to avoid sign/zero collapse',
    )
    parser.add_argument(
        '--delta_mul_coeff_max',
        type=float,
        default=3.0,
        help='upper bound for multiplicative coefficient in mul_z fusion',
    )
    parser.add_argument(
        '--kernel_amp_bins',
        type=int,
        default=21,
        help='number of amplitude bins for kernel token mode; default supports AMP_0..AMP_20',
    )
    parser.add_argument(
        '--kernel_rel_norm_thresh',
        type=float,
        default=0.05,
        help='residual norm threshold for REL=0 in kernel fitter',
    )
    parser.add_argument(
        '--kernel_rel_improve_ratio',
        type=float,
        default=0.0,
        help='minimum relative SSE improvement (vs zero-kernel) required to keep REL=1',
    )
    parser.add_argument(
        '--kernel_rel_improve_abs',
        type=float,
        default=0.0,
        help='minimum absolute SSE improvement (vs zero-kernel) required to keep REL=1',
    )
    parser.add_argument(
        '--kernel_a_max',
        type=float,
        default=2.0,
        help='max projected amplitude in kernel fitter',
    )
    parser.add_argument(
        '--kernel_sft_lr',
        type=float,
        default=1e-4,
        help='learning rate for kernel token SFT training',
    )
    parser.add_argument(
        '--kernel_gen_max_new_tokens',
        type=int,
        default=32,
        help='max generated tokens for kernel parameter sequence at inference',
    )
    parser.add_argument(
        '--kernel_api_enable',
        type=int,
        default=0,
        choices=[0, 1],
        help='enable API-assisted relabeling for uncertain kernel SFT samples',
    )
    parser.add_argument(
        '--kernel_api_key',
        type=str,
        default='',
        help='API key for kernel relabeling; falls back to OPENAI_API_KEY when empty',
    )
    parser.add_argument(
        '--kernel_api_model',
        type=str,
        default='gpt-5.1',
        help='OpenAI model for kernel relabeling (dataset stage uses fixed chatgpt-5.1)',
    )
    parser.add_argument(
        '--kernel_api_temperature',
        type=float,
        default=0.1,
        help='temperature for API relabeling calls',
    )
    parser.add_argument(
        '--kernel_api_max_calls',
        type=int,
        default=200,
        help='max API calls during one kernel sample build; <0 means unlimited',
    )
    parser.add_argument(
        '--kernel_api_uncertain_band',
        type=float,
        default=0.02,
        help='query API when rel_norm is within this band around kernel_rel_norm_thresh',
    )
    parser.add_argument(
        '--kernel_api_low_amp_bin',
        type=int,
        default=2,
        help='query API when auto label is REL=1 but AMP bin is <= this threshold',
    )
    parser.add_argument(
        '--kernel_api_log_every',
        type=int,
        default=10,
        help='log API relabeling progress every N calls',
    )
    parser.add_argument(
        '--kernel_api_log_examples',
        type=int,
        default=3,
        help='log up to N API response examples for success/failure at dataset build end',
    )
    parser.add_argument(
        '--kernel_api_live_fail_log_max',
        type=int,
        default=3,
        help='log up to N live API failure samples during dataset build',
    )
    parser.add_argument(
        '--kernel_api_price_in_per_1m',
        type=float,
        default=5.0,
        help='estimated USD price per 1M input tokens for API cost logging',
    )
    parser.add_argument(
        '--kernel_api_price_out_per_1m',
        type=float,
        default=15.0,
        help='estimated USD price per 1M output tokens for API cost logging',
    )
    parser.add_argument(
        '--kernel_cache_file',
        type=str,
        default='sft_kernel_cache.json',
        help='cache file name for kernel-token SFT samples under checkpoints/<taskName>/',
    )
    parser.add_argument(
        '--kernel_amp_table_file',
        type=str,
        default='kernel_amp_table.json',
        help='amp table file name under checkpoints/<taskName>/ for kernel-token mode',
    )
    parser.add_argument(
        '--kernel_api_cache_file',
        type=str,
        default='sft_kernel_api_cache.json',
        help='API relabel cache file name under checkpoints/<taskName>/',
    )
    parser.add_argument(
        '--kernel_api_type',
        type=str,
        default='both',
        choices=['priors', 'relsign', 'both'],
        help='dataset-stage API mode: priors-only, relsign-only, or both',
    )
    parser.add_argument(
        '--kernel_api_prior_rel_norm_thresh',
        type=float,
        default=-1.0,
        help='trigger priors API when rel_norm >= this value; <=0 uses kernel_rel_norm_thresh',
    )
    parser.add_argument('--utility_rank_lambda', type=float, default=0.2,
                        help='weight for real-vs-null utility ranking loss on rel logits')
    parser.add_argument('--utility_rank_margin', type=float, default=0.10,
                        help='margin for ranking loss: rel_real - rel_null should exceed this value')


    parser.add_argument("--patch_dropout", type=float, default=0.0)
    parser.add_argument("--head_dropout", type=float, default=0.0)
    parser.add_argument("--head_mlp", action="store_true", default=False)

    parser.add_argument("--policy_space", type=list, default=["all"])
    parser.add_argument("--default_policy", type=str, default="smart", help="default news selection policy for training and evaluation")
    parser.add_argument("--smart_rel_weight", type=float, default=0.55, help="weight of semantic relevance in smart news retrieval")
    parser.add_argument("--smart_kw_weight", type=float, default=0.15, help="weight of keyword coverage in smart news retrieval")
    parser.add_argument("--smart_rate_weight", type=float, default=0.15, help="weight of external news rating in smart news retrieval")
    parser.add_argument("--smart_recency_weight", type=float, default=0.15, help="weight of recency prior in smart news retrieval")
    parser.add_argument("--smart_recency_tau", type=float, default=8.0, help="rank-decay temperature for recency in smart retrieval")
    parser.add_argument("--smart_mmr_lambda", type=float, default=0.75, help="MMR relevance-vs-diversity tradeoff in smart retrieval")
    parser.add_argument("--smart_dedup_threshold", type=float, default=0.92, help="cosine threshold for near-duplicate news suppression")
    parser.add_argument("--utility_rerank_enable", type=int, default=1, choices=[0, 1],
                        help="rerank selected news by utility score before prompt composition")
    parser.add_argument("--utility_keyword_weight", type=float, default=0.35, help="utility score weight: keyword coverage")
    parser.add_argument("--utility_recency_weight", type=float, default=0.25, help="utility score weight: recency")
    parser.add_argument("--utility_rate_weight", type=float, default=0.35, help="utility score weight: external rate column")
    parser.add_argument("--utility_sentiment_weight", type=float, default=0.05, help="utility score weight: sentiment magnitude")
    parser.add_argument("--utility_recency_tau_hours", type=float, default=24.0, help="time-decay tau in hours for utility rerank")
    parser.add_argument("--utility_mmr_enable", type=int, default=1, choices=[0, 1], help="enable MMR diversification in utility rerank")
    parser.add_argument("--utility_mmr_lambda", type=float, default=0.8, help="MMR lambda in utility rerank")
    parser.add_argument("--utility_dedup_threshold", type=float, default=0.95, help="dedup threshold in utility rerank")
    parser.add_argument("--utility_keep_topk", type=int, default=-1, help="optional post-rerank truncation; <=0 disables")
    parser.add_argument("--utility_min_score", type=float, default=-1.0, help="drop selected news below this utility score; <0 disables")
    parser.add_argument("--utility_show_in_prompt", type=int, default=1, choices=[0, 1],
                        help="show utility score tag for each news item in prompt")

    #TIME-SEIRES DATA PATCH LEN
    parser.add_argument("--patch_len", type=int, default=4)
    # ==== News dropout ====
    parser.add_argument("--news_dropout", type=float, default=0.0)

    parser.add_argument("--stage", type=str, default="delta",
                        choices=["all", "base", "delta"],
                        help="Run stage: base only, delta only, or all (base->delta).")
    #默认是把epochs按比例分配，这里可以写死
    parser.add_argument("--base_epochs", type=int, default=-1,
                    help="If >=0, override base epochs when stage=all or stage=base.")

    parser.add_argument("--delta_epochs", type=int, default=4,
                        help="If >=0, override delta epochs when stage=all or stage=delta.")

    # ==== pure TS base backbone (scheme2) ====
    parser.add_argument("--base_backbone", type=str, default="dlinear", choices=["dlinear", "mlp"],
                        help="pure TS backbone used in base stage")
    parser.add_argument("--base_hidden_dim", type=int, default=256,
                        help="hidden dim for MLP base backbone")
    parser.add_argument("--base_moving_avg", type=int, default=25,
                        help="moving average kernel for DLinear decomposition")
    parser.add_argument("--base_dropout", type=float, default=0.0,
                        help="dropout in base backbone")
    parser.add_argument("--base_lr", type=float, default=-1.0,
                        help="learning rate for base backbone (<=0 to reuse --lr)")
    parser.add_argument("--base_weight_decay", type=float, default=-1.0,
                        help="weight decay for base backbone (<0 to reuse --weight_decay)")
    parser.add_argument("--base_loss", type=str, default="smooth_l1", choices=["mse", "mae", "smooth_l1"],
                        help="point loss used in base stage training/eval")
    # ==== residual loss =====
    parser.add_argument("--residual_loss", type=str, default="mse", choices = ['mse', 'mae', 'smooth_l1'])
    parser.add_argument("--residual_base_frac",type=float,default=0.4)
    # ===== Basic =====
    parser.add_argument('--taskName', type=str, default='task1', help='The name of this running task')
    parser.add_argument('--gpu', type=int, default=0, help='gpu id')
    parser.add_argument('--seed', type=int, default=42, help='random seed')

    # ===== Data & Instruction =====
    parser.add_argument('--dayFirst', action="store_true", default=False, help='Are your datasets in "day-first" format?')
    parser.add_argument('--train_file', type=str, default='', help='train set path (CSV/Parquet)')
    parser.add_argument('--val_file', type=str, default='', help='validation set path')
    parser.add_argument('--test_file', type=str, default='', help='test set path')
    parser.add_argument('--time_col', type=str, default='timestamp', help='time column name')
    parser.add_argument('--value_col', type=str, default='value', help='target/series column name')
    parser.add_argument('--id_col', type=str, default=None, help='optional series ID column for multi-series')
    parser.add_argument('--freq_min', type=int, default=30, help='sampling frequency in minutes (e.g., 30 for half-hour)')
    parser.add_argument('--region', type=str, default='', help='region/state/country code')
    parser.add_argument('--unit', type=str, default='', help='unit string')
    parser.add_argument('--volatility_bin_tiers', type=int, default=100, help='the tiers to bin volatility')
    parser.add_argument('--token_budget', type=int, default=700, help='max tokens for composed prompt')
    parser.add_argument('--val_ema_alpha', type=float, default=0.9, help='EMA alpha for validation loss smoothing')

    # Whether to include explanations in the prompt template
    parser.add_argument('--need_explain', action='store_true', help='include explanations in template')
    parser.add_argument('--need_ci', action='store_true', help='include confidence interval in output')

    # ===== Task windowing =====
    parser.add_argument('--history_len', type=int, default=48, help='steps for history window L')
    parser.add_argument('--horizon', type=int, default=48, help='steps to predict H')
    parser.add_argument('--stride', type=int, default=48, help='sliding stride for training')
    # ===== News retrieval (rule-based) =====
    parser.add_argument('--news_path', type=str, default='', help='path to news store (should be a JSON file)')
    parser.add_argument('--news_time_col', type=str, default='date', help='news timestamp column name')
    parser.add_argument('--news_text_col', type=str, default='content', help='news text/summary column name')
    parser.add_argument('--news_tz', type=str, default='', help='timezone for news timestamps')
    parser.add_argument('--news_window_days', type=int, default=1, help='look-back window (days) before target time')
    parser.add_argument('--news_topM', type=int, default=20, help='candidate news cap per sample')
    parser.add_argument('--news_topK', type=int, default=5, help='news K after policy/RL')
    
    # ===== News summarization =====
    # News is pre-summarized offline in this workflow, so this option is usually unused.
    parser.add_argument('--news_summary_method', type=str, default='none', choices=['none', 'lead3', 'rule'],
                        help='shorten news before inserting to prompt')
    parser.add_argument('--news_max_sentences', type=int, default=3, help='max sentences per selected news')



    # ===== Prompt templates =====
    parser.add_argument('--template_pool', type=str, default='configs/templates3.yaml',
                        help='YAML/JSON templates with placeholders')

    #===== Token budget fractions =====
    parser.add_argument('--token_budget_news_frac', type=float, default=0.9, help='budget frac for news')

    # ===== LLaMA =====
    parser.add_argument('--base_model', type=str, default='meta-llama/Meta-Llama-3-8B', help='HF model id or local path')
    parser.add_argument('--tokenizer', type=str, default='', help='HF tokenizer id (default: same as base_model)')
    parser.add_argument('--load_in_4bit', action='store_true', help='use 4-bit quantization (QLoRA)')
    parser.add_argument('--gradient_checkpointing', action='store_true', help='enable gradient checkpointing')
    parser.add_argument('--max_seq_len', type=int, default=768, help='max sequence length')

    # LoRA hyperparameters
    parser.add_argument('--lora_r', type=int, default=8, help='LoRA rank')
    parser.add_argument('--lora_alpha', type=int, default=32, help='LoRA alpha')
    parser.add_argument('--lora_dropout', type=float, default=0.05, help='LoRA dropout')
    parser.add_argument('--target_modules', type=str, default='q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj',
                        help='comma-separated target module names for LoRA')
    parser.add_argument('--lr', type=float, default=5e-6, help='learning rate for LoRA params')
    parser.add_argument('--scheduler', type=int, default=1, help='1 =on; 0 =off for lr scheduler')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='weight decay')
    parser.add_argument('--warmup_ratio', type=float, default=0.03, help='warmup ratio')
    parser.add_argument('--batch_size', type=int, default=2, help='micro batch size per device')
    parser.add_argument('--grad_accum', type=int, default=16, help='gradient accumulation steps')
    parser.add_argument('--epochs', type=int, default=3, help='outer epochs over the dataset')
    parser.add_argument('--save_dir', type=str, default='./checkpoints', help='checkpoint dir')

    # ===== RL / Bandit =====
    parser.add_argument('--rl_use', type=int, default=0, help='use RL/bandit for news selection? (0/1)')
    parser.add_argument('--rl_algo', type=str, default='lints', choices=['lints','linucb'], help='bandit algorithm')

    parser.add_argument('--reward_metric', type=str, default='mae', choices=['mse','mae','loss'], help='reward metric')
    parser.add_argument('--reward_ema', type=float, default=0.3, help='EMA smoothing of reward')
    parser.add_argument('--domain_reward_norm', action='store_true', help='z-score reward per (domain,horizon) group')
    parser.add_argument('--ucb_alpha', type=float, default=1.0, help='LinUCB alpha')
    parser.add_argument('--ts_v', type=float, default=1.0, help='LinTS prior scale v')
    parser.add_argument('--epsilon', type=float, default=0.05, help='epsilon-greedy fallback')
    parser.add_argument('--news_rl_enable', type=int, default=1, choices=[0, 1],
                        help='enable contextual bandit for per-prompt news item selection in delta stage')
    parser.add_argument('--news_rl_algo', type=str, default='auto', choices=['auto', 'lints', 'linucb'],
                        help='algo for news bandit; auto follows --rl_algo')
    parser.add_argument('--news_rl_k_choices', type=str, default='1,2,3,5,7,10',
                        help='candidate K values for RL to choose per prompt')
    parser.add_argument('--news_rl_allow_over_topk', type=int, default=0, choices=[0, 1],
                        help='allow RL-selected K to exceed --news_topK')
    parser.add_argument('--news_rl_epsilon', type=float, default=0.05,
                        help='epsilon-greedy exploration for news item/K selection')
    parser.add_argument('--news_rl_prefilter_mult', type=int, default=4,
                        help='prefilter pool size multiplier over max(K) before RL item selection')
    parser.add_argument('--news_rl_pool_cap', type=int, default=128,
                        help='max candidate pool size for RL item selection')
    parser.add_argument('--news_rl_reward_clip', type=float, default=3.0,
                        help='clip absolute reward when updating news bandits')
    parser.add_argument('--news_rl_recency_tau_hours', type=float, default=24.0,
                        help='recency decay tau (hours) in news item features for RL')

    # ===== Eval & Logging =====
    parser.add_argument('--early_stop_patience', type=int, default=4, help='patience in eval rounds')
    parser.add_argument(
        '--delta_val_mode',
        type=str,
        default='each_epoch',
        choices=['each_epoch', 'end_only', 'none'],
        help='delta-stage validation schedule: each epoch, end only, or disabled',
    )
    parser.add_argument('--eval_progress_bar', type=int, default=1, choices=[0, 1],
                        help='show tqdm progress bar during validation/testing evaluation loops')
    parser.add_argument('--eval_progress_leave', type=int, default=0, choices=[0, 1],
                        help='keep evaluation progress bars after completion')

    # ===== Metadata =====
    parser.add_argument("--description",type=str,default="",help="描述这个 dataset 的用途，例如 '新州电价数据'")
    parser.add_argument("--select_policy_by",type = str,default = "epoch",choices=["epoch", "batch"], 
                        help = "Select policy/template by epoch-level or batch-level"
    )

    args = parser.parse_args()

    if not args.tokenizer:
        args.tokenizer = args.base_model
    args.target_modules = [s.strip() for s in args.target_modules.split(',') if s.strip()]
    args.token_budget_news_frac = float(max(0.0, min(1.0, args.token_budget_news_frac)))

    main_train(args)
