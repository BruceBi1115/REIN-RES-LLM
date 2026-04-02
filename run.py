# run.py
import argparse
# from src.trainer import main as main_train
from src.base_delta_decoouple_trainer import main as main_train
# from src.original_residual_trainer import main as main_train

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Cross-domain Forecasting')

    
    parser.add_argument('--rel_lambda', type=float, default=0.3, help='weight for relative loss')
    parser.add_argument('--delta_sign_eps', type=float, default=0.03,
                        help='additive mode: ignore sign supervision where |delta_target| <= eps; relative mode: neutral band for q-state labels')
    parser.add_argument('--delta_sign_tau', type=float, default=1.0,
                        help='temperature for additive-mode internal soft sign = tanh(sign_logits / tau)')
    parser.add_argument('--delta_sign_mode', type=str, default='signnet_binary', choices=['signnet_binary', 'internal'],
                        help='residual-state mode for DELTA: external SignNet state classifier or internal DELTA state head')
    parser.add_argument('--delta_sign_external_epochs', type=int, default=60,
                        help='epochs for external signnet pretraining when delta_sign_mode=signnet_binary')
    parser.add_argument('--delta_sign_external_hidden', type=int, default=128,
                        help='hidden size for external signnet')
    parser.add_argument('--delta_sign_external_dropout', type=float, default=0.2,
                        help='dropout for external signnet')
    parser.add_argument('--delta_sign_external_lr', type=float, default=3e-4,
                        help='learning rate for external signnet pretraining')
    parser.add_argument('--delta_sign_external_weight_decay', type=float, default=5e-4,
                        help='weight decay for external signnet pretraining')
    parser.add_argument('--delta_sign_external_grad_clip', type=float, default=1.0,
                        help='grad clip for external signnet pretraining')
    parser.add_argument('--delta_sign_external_patience', type=int, default=10,
                        help='early-stop patience for external signnet validation')
    parser.add_argument('--delta_sign_external_select_metric', type=str, default='acc', choices=['acc', 'balanced_acc', 'loss'],
                        help='model-selection metric for external signnet early stopping')
    parser.add_argument('--delta_sign_external_min_delta', type=float, default=1e-4,
                        help='minimum validation loss improvement required to reset early-stop counter')
    parser.add_argument('--delta_sign_external_lr_factor', type=float, default=0.5,
                        help='ReduceLROnPlateau factor for external signnet (set >=1 to disable)')
    parser.add_argument('--delta_sign_external_lr_patience', type=int, default=1,
                        help='ReduceLROnPlateau patience (epochs) for external signnet')
    parser.add_argument('--delta_sign_external_min_lr', type=float, default=1e-5,
                        help='minimum learning rate for external signnet scheduler')
    parser.add_argument('--delta_sign_external_calibrate_bias', type=int, default=1, choices=[0, 1],
                        help='calibrate external signnet decision bias on validation set after pretraining (additive mode only)')
    parser.add_argument('--delta_sign_external_bias_clip', type=float, default=2.0,
                        help='absolute clip bound for calibrated decision bias')
    parser.add_argument('--delta_sign_external_news_dropout', type=int, default=0, choices=[0, 1],
                        help='apply news dropout during external signnet training (0 recommended for stability)')
    parser.add_argument('--delta_sign_external_use_news_weighting', type=int, default=0, choices=[0, 1],
                        help='apply news usefulness weights in external signnet supervision loss')
    parser.add_argument('--delta_sign_external_use_residual_weighting', type=int, default=0, choices=[0, 1],
                        help='apply residual-magnitude position weights in external signnet supervision loss')
    parser.add_argument('--delta_sign_external_use_pos_weight', type=int, default=1, choices=[0, 1],
                        help='enable masked dynamic pos_weight for additive-mode external signnet BCE')
    parser.add_argument('--delta_sign_external_pos_weight_floor', type=float, default=0.5,
                        help='lower bound for additive-mode dynamic pos_weight in external signnet BCE')
    parser.add_argument('--delta_sign_external_pos_weight_clip', type=float, default=3.0,
                        help='upper bound for additive-mode dynamic pos_weight in external signnet BCE')
    parser.add_argument('--delta_sign_external_tau', type=float, default=1.0,
                        help='temperature for mapping external additive-mode signnet logits to soft sign via tanh')
    parser.add_argument('--delta_mag_target', type=str, default='log1p', choices=['raw', 'log1p'],
                        help='target transform used for magnitude regression')
    parser.add_argument('--delta_mag_max', type=float, default=0.0,
                        help='optional clamp for predicted magnitude in z-space; <=0 disables')
    parser.add_argument('--delta_residual_weight_scale', type=float, default=1.0,
                        help='bounded extra weight for larger true residual positions')
    parser.add_argument('--news_usefulness_weighting', type=int, default=1, choices=[0, 1],
                        help='apply bounded per-sample usefulness weighting to news-specific losses')
    parser.add_argument('--delta_alpha_scale', type=float, default=0.75,
                        help='maximum multiplicative deformation range for alpha_news: 1 + scale * tanh(logits)')
    parser.add_argument('--delta_patch_prototypes', type=int, default=0,
                        help='optional number of learnable patch prototypes for delta patch reprogramming; 0 disables')
    parser.add_argument('--delta_patch_proto_temp', type=float, default=1.0,
                        help='softmax temperature for optional delta patch prototype routing')
    parser.add_argument('--doc_candidate_mode', type=str, default='beta_only',
                        choices=['beta_only'],
                        help='document candidate construction mode for doc impact aggregation')

    parser.add_argument('--delta_head_init_std', type=float, default=0.01, help='std for delta head weight init')
    parser.add_argument('--delta_clip', type=float, default=3.0, help='tanh clip for delta outputs in z-space (<=0 to disable)')
    parser.add_argument('--delta_news_tail_tokens', type=int, default=160, help='how many tail text tokens to pool as news context')
    parser.add_argument('--delta_model_variant', type=str, default='tiny_news_ts', choices=['tiny_news_ts'],
                        help='DELTA model branch')
    parser.add_argument('--tiny_news_hidden_size', type=int, default=256,
                        help='hidden size for DELTA tiny_news_ts branch')

    parser.add_argument('--delta_grad_clip', type=float, default=1.0, help='grad clip norm for delta stage (<=0 to disable)')
    parser.add_argument('--delta_head_lr_scale', type=float, default=1.0, help='lr scale for delta/rel heads in delta stage')
    parser.add_argument('--delta_other_lr_scale', type=float, default=0.5, help='lr scale for other trainable params in delta stage')
    parser.add_argument('--delta_freeze_feature_modules', type=int, default=0, choices=[0, 1],
                        help='freeze patch/pooling feature modules in delta stage (legacy behavior)')
    parser.add_argument('--delta_target_clip', type=float, default=0.0,
                        help='optional clip for delta_target = target-base_pred in z-space; <=0 disables')
    parser.add_argument('--delta_residual_mode', type=str, default='additive', choices=['additive', 'relative'],
                        help='delta branch target/fusion mode: additive residual or relative percentage residual')
    parser.add_argument('--delta_relative_denom_floor', type=float, default=0.0,
                        help='minimum absolute scale floor (raw scale) when computing relative percentage residual q')
    parser.add_argument('--delta_relative_ratio_clip', type=float, default=0.0,
                        help='optional clip for relative percentage residual q; <=0 disables clipping')
    parser.add_argument('--cleaned_residual_enable', type=int, default=1, choices=[0, 1],
                        help='use cleaned residual targets for SignNet and DELTA auxiliary supervision')
    parser.add_argument('--cleaned_residual_smooth_alpha', type=float, default=0.6,
                        help='EWMA alpha for horizon-wise cleaned residual smoothing; higher means smoother')
    parser.add_argument('--cleaned_residual_structured_mix', type=float, default=0.35,
                        help='blend weight for structured-news residual template in cleaned residual construction')
    parser.add_argument('--news_refine_mode', type=str, default='local', choices=['local', 'api'],
                        help='news refinement backend; local is join+truncate fallback')
    parser.add_argument('--news_refine_cache_enable', type=int, default=1, choices=[0, 1],
                        help='enable persistent cache for refined news text')
    parser.add_argument('--news_doc_cache_path', type=str, default='',
                        help='optional unified refined-news cache path; when present, this file is treated as the primary cache object store')
    parser.add_argument('--news_doc_cache_explicit', type=int, default=0, choices=[0, 1],
                        help='whether the unified cache path was explicitly requested by the caller rather than auto-discovered')
    parser.add_argument('--news_refine_cache_path', type=str, default='',
                        help='optional cache file path for refined news; default is a shared cache under checkpoints/_shared_refine_cache/')
    parser.add_argument('--news_refine_cache_read_path', type=str, default='',
                        help='optional existing refined-news cache path(s) to preload before this run; supports comma-separated paths')
    parser.add_argument('--news_structured_cache_enable', type=int, default=1, choices=[0, 1],
                        help='enable persistent cache for structured news labels')
    parser.add_argument('--news_structured_cache_path', type=str, default='',
                        help='optional cache file path for structured news labels; default is auto-generated from news filename')
    parser.add_argument('--news_structured_cache_read_path', type=str, default='',
                        help='optional existing structured-news cache path(s) to preload before this run; supports comma-separated paths')
    parser.add_argument('--news_refine_prewarm', type=int, default=1, choices=[0, 1],
                        help='prewarm refine cache with one train split pass before DELTA training')
    parser.add_argument('--news_refine_prewarm_max_batches', type=int, default=-1,
                        help='limit prewarm news documents; <=0 means all in-scope news documents')
    parser.add_argument('--news_refine_show_progress', type=int, default=1, choices=[0, 1],
                        help='show tqdm progress bar in terminal during refine cache prewarm')
    parser.add_argument('--news_structured_prewarm', type=int, default=1, choices=[0, 1],
                        help='prewarm structured cache alongside refine preprocessing before DELTA training')
    parser.add_argument('--news_structured_show_progress', type=int, default=1, choices=[0, 1],
                        help='show tqdm progress bar in terminal during structured cache prewarm')
    parser.add_argument('--news_structured_mode', type=str, default='off', choices=['off', 'heuristic', 'api'],
                        help='structured event extraction backend')
    parser.add_argument('--delta_structured_enable', type=int, default=0, choices=[0, 1],
                        help='allow DELTA to consume structured event features directly')
    parser.add_argument('--delta_structured_feature_dim', type=int, default=12,
                        help='feature dimension for structured event vector injected into DELTA')
    parser.add_argument('--delta_temporal_text_enable', type=int, default=0, choices=[0, 1],
                        help='enable time-aligned refined-text auxiliary sequence for DELTA')
    parser.add_argument('--delta_temporal_text_source', type=str, default='refined', choices=['refined', 'raw'],
                        help='source used by the temporal-text branch: refined cached snippets or original raw news text')
    parser.add_argument('--temporal_text_model_id', type=str, default='',
                        help='optional HF model id or local path for the temporal-text encoder/tokenizer; defaults to --tokenizer or --base_model')
    parser.add_argument('--delta_temporal_text_dim', type=int, default=8,
                        help='projected feature dimension for the time-aligned text auxiliary sequence')
    parser.add_argument('--delta_temporal_text_max_len', type=int, default=96,
                        help='maximum token length for each time-aligned auxiliary text snippet')
    parser.add_argument('--delta_temporal_text_per_step_topk', type=int, default=3,
                        help='maximum number of refined news snippets merged for each history step')
    parser.add_argument('--delta_temporal_text_fuse_lambda', type=float, default=0.5,
                        help='strength of patch-level fusion from time-aligned text auxiliary features into DELTA')
    parser.add_argument('--delta_temporal_text_freeze_encoder', type=int, default=1, choices=[0, 1],
                        help='freeze the pretrained text encoder used for the time-aligned text auxiliary sequence')
    parser.add_argument('--delta_multimodal_arch', type=str, default='summary_gated', choices=['summary_gated', 'plan_c_mvp'],
                        help='residual architecture: legacy summary/gated branch or Plan-C-style regime router + expert mixture MVP')
    parser.add_argument('--delta_multimodal_fuse_lambda', type=float, default=1.0,
                        help='mixture strength for Plan C regime experts in DELTA and external SignNet')
    parser.add_argument('--delta_route_balance_lambda', type=float, default=0.02,
                        help='Plan C expert-route load balancing regularization weight; <=0 disables')
    parser.add_argument('--delta_route_abstain_lambda', type=float, default=0.05,
                        help='Plan C abstain regularization weight when usable news exists; <=0 disables')
    parser.add_argument('--delta_route_abstain_target', type=float, default=0.35,
                        help='Plan C abstain probability target before penalty activates')
    parser.add_argument('--delta_route_conf_floor', type=float, default=0.25,
                        help='minimum confidence multiplier retained after Plan C abstain gating')
    parser.add_argument('--news_api_enable', type=int, default=0, choices=[0, 1],
                        help='caller-level switch recording whether API-backed news processing was requested')
    parser.add_argument('--news_api_model', type=str, default='gpt-5.1',
                        help='OpenAI model name used by API adapter when news_*_mode=api')
    parser.add_argument('--news_api_key_path', type=str, default='api_key.txt',
                        help='path to API key file (used if OPENAI_API_KEY is not set)')
    parser.add_argument('--news_api_base_url', type=str, default='',
                        help='optional custom OpenAI-compatible base URL')
    parser.add_argument('--news_api_timeout_sec', type=float, default=30.0,
                        help='timeout for one API request in seconds')
    parser.add_argument('--news_api_max_retries', type=int, default=2,
                        help='max retries for one API request')
    parser.add_argument('--delta_include_structured_news', type=int, default=0, choices=[0, 1],
                        help='append structured event fields to delta prompt news context')
    parser.add_argument('--hard_reflection_mode', type=str, default='off', choices=['off', 'api'],
                        help='optional hard-sample reflection backend (offline utility hook)')
    parser.add_argument('--hard_reflection_topk', type=int, default=8,
                        help='number of hardest samples per epoch sent to reflection hook')
    parser.add_argument("--patch_dropout", type=float, default=0.0)
    parser.add_argument("--head_dropout", type=float, default=0.0)
    parser.add_argument("--head_mlp", action="store_true", default=False)

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
    parser.add_argument('--dayFirst', action="store_true", default=True, help='Are your datasets in "day-first" format?')
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
    parser.add_argument('--news_topK', type=int, default=5, help='news K after policy')
    
    # ===== Prompt templates =====
    parser.add_argument('--template_pool', type=str, default='configs/templates3.yaml',
                        help='YAML/JSON templates with placeholders')

    #===== Token budget fractions =====
    parser.add_argument('--token_budget_news_frac', type=float, default=0.9, help='budget frac for news')

    # ===== Text Model Tokenizer =====
    parser.add_argument('--base_model', type=str, default='distilbert-base-uncased', help='HF model id or local path')
    parser.add_argument('--tokenizer', type=str, default='', help='HF tokenizer id (default: same as base_model)')
    parser.add_argument('--max_seq_len', type=int, default=768, help='max sequence length')

    parser.add_argument('--lr', type=float, default=5e-6, help='learning rate')
    parser.add_argument('--scheduler', type=int, default=1, help='1 =on; 0 =off for lr scheduler')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='weight decay')
    parser.add_argument('--warmup_ratio', type=float, default=0.03, help='warmup ratio')
    parser.add_argument('--batch_size', type=int, default=2, help='micro batch size per device')
    parser.add_argument('--grad_accum', type=int, default=16, help='gradient accumulation steps')
    parser.add_argument('--epochs', type=int, default=3, help='outer epochs over the dataset')
    parser.add_argument('--save_dir', type=str, default='./checkpoints', help='checkpoint dir')

    parser.add_argument('--select_metric', type=str, default='mae', choices=['mse','mae','loss'],
                        help='metric used for model selection and comparisons')

    # ===== Eval & Logging =====
    parser.add_argument('--early_stop_patience', type=int, default=5, help='patience in eval rounds')
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
    args = parser.parse_args()

    if not args.tokenizer:
        args.tokenizer = args.base_model
    args.token_budget_news_frac = float(max(0.0, min(1.0, args.token_budget_news_frac)))

    main_train(args)
