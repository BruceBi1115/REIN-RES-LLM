# run.py
import argparse
import pandas as pd
# from src.trainer import main as main_train
from src.base_delta_decoouple_trainer import main as main_train
# from src.original_residual_trainer import main as main_train
from src.chatgpt_4o_mini_keyword_generate.gpt_client import run_from_config, stream_from_config
from pathlib import Path
from openai import OpenAI
from textblob import TextBlob

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Cross-domain LLaMA Forecasting with RL')

    
    parser.add_argument('--rel_lambda', type=float, default=0.3, help='weight for relative loss')
    parser.add_argument('--rel_supervise_lambda', type=float, default=0.5,
                        help='weight for supervised rel_head BCE in delta stage')
    parser.add_argument("--rel_on", action="store_true", help="whether to use relative loss")

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
    parser.add_argument('--delta_auto_alpha', type=int, default=0, choices=[0, 1],
                        help='deprecated: alpha fusion is disabled; kept for backward compatibility')
    parser.add_argument('--delta_alpha_candidates', type=str, default='1.0',
                        help='deprecated: alpha fusion is disabled; kept for backward compatibility')
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
    parser.add_argument('--precision', type=str, default='bf16', choices=['fp32', 'fp16', 'bf16'],
                        help='training precision/mixed precision mode')

    # ===== Data & Instruction =====
    parser.add_argument('--data_root', type=str, default='./dataset', help='root folder for datasets')
    parser.add_argument('--dayFirst', action="store_true", default=False, help='Are your datasets in "day-first" format?')
    parser.add_argument('--train_file', type=str, default='', help='train set path (CSV/Parquet)')
    parser.add_argument('--val_file', type=str, default='', help='validation set path')
    parser.add_argument('--test_file', type=str, default='', help='test set path')
    parser.add_argument('--time_col', type=str, default='timestamp', help='time column name')
    parser.add_argument('--value_col', type=str, default='value', help='target/series column name')
    parser.add_argument('--id_col', type=str, default=None, help='optional series ID column for multi-series')
    parser.add_argument('--scaler', type=str, default='standard', choices=['none', 'standard', 'minmax'],
                        help='scaling method for target')

    parser.add_argument('--instruction_json', type=str, default='',
                        help='path to JSON with instruction fields per run/domain')
    parser.add_argument('--freq_min', type=int, default=30, help='sampling frequency in minutes (e.g., 30 for half-hour)')
    parser.add_argument('--region', type=str, default='', help='region/state/country code')
    parser.add_argument('--unit', type=str, default='', help='unit string')
    parser.add_argument('--season', type=str, default='', help='DJF/MAM/JJA/SON or empty')
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
    parser.add_argument('--target_precision', type=int, default=3, help='decimal places for target values')
    # ===== News retrieval (rule-based) =====
    parser.add_argument('--news_path', type=str, default='', help='path to news store (should be a JSON file)')
    parser.add_argument('--news_time_col', type=str, default='date', help='news timestamp column name')
    parser.add_argument('--news_text_col', type=str, default='content', help='news text/summary column name')
    parser.add_argument('--news_source_col', type=str, default='source', help='news source column (optional)')
    parser.add_argument('--news_tz', type=str, default='', help='timezone for news timestamps')
    parser.add_argument('--news_window_days', type=int, default=1, help='look-back window (days) before target time')
    parser.add_argument('--news_topM', type=int, default=20, help='candidate news cap per sample')
    parser.add_argument('--news_topK', type=int, default=5, help='news K after policy/RL')
    # parser.add_argument('--news_policy', type=str, default='',
    #                     help='rule-based extraction strategy or combinational bandit')
    
    # Keyword files for policy-based news selection
    parser.add_argument('--keyword_path', type=str, default='keywords/kws.txt',
                        help='keyword list for filtering news (one per line)')
    
    # ===== News summarization =====
    # WE ALREADY USED CHATGPT4O TO SUMMARIZE NEWS, SO THIS IS NOT USED
    parser.add_argument('--news_summary_method', type=str, default='none', choices=['none', 'lead3', 'rule'],
                        help='shorten news before inserting to prompt')
    parser.add_argument('--news_max_sentences', type=int, default=3, help='max sentences per selected news')



    # ===== Prompt templates =====
    parser.add_argument('--template_pool', type=str, default='configs/templates3.yaml',
                        help='YAML/JSON templates with placeholders')
    # Use template_ids to restrict to a subset of templates
    parser.add_argument('--template_ids', type=str, default='', help='comma-separated template ids to allow (empty=all)')

    #===== Token budget fractions =====
    parser.add_argument('--token_budget_history_frac', type=float, default=0.05, help='budget frac for history')
    parser.add_argument('--token_budget_news_frac', type=float, default=0.9, help='budget frac for news')
    parser.add_argument('--token_budget_instr_frac', type=float, default=0.05, help='budget frac for instruction')

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
    parser.add_argument('--max_steps', type=int, default=-1, help='override total steps if >0')
    # parser.add_argument('--eval_interval', type=int, default=200, help='validate every N steps')
    parser.add_argument('--save_dir', type=str, default='./checkpoints', help='checkpoint dir')
    parser.add_argument('--save_interval', type=int, default=1000, help='save every N steps')

    # ===== RL / Bandit =====
    parser.add_argument('--rl_use', type=int, default=0, help='use RL/bandit for news selection? (0/1)')
    parser.add_argument('--rl_algo', type=str, default='lints', choices=['lints','linucb'], help='bandit algorithm')
    parser.add_argument('--rl_cycle_steps', type=int, default=100, help='short SFT steps per decision cycle (0=no-train)')
    parser.add_argument('--rl_update_times', type=int, default=1, help='update bandits every N cycles')
    parser.add_argument('--rl_val_probe_size', type=int, default=256, help='fixed validation probe size')
    parser.add_argument('--rl_val_probe_frac', type=float, default=0.5, help='fixed validation probe fraction')

    parser.add_argument('--reward_metric', type=str, default='mae', choices=['mse','mae','loss'], help='reward metric')
    parser.add_argument('--reward_mode', type=str, default='delta', choices=['delta','negative'], help='delta or negative')
    parser.add_argument('--reward_len_penalty', type=float, default=0.0, help='penalty for prompt tokens')
    parser.add_argument('--reward_k_penalty', type=float, default=0.0, help='penalty for K news')
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
    parser.add_argument('--early_stop_patience', type=int, default=5, help='patience in eval rounds')
    # parser.add_argument('--log_dir', type=str, default='./logs', help='log directory')
    # parser.add_argument('--run_name', type=str, default='xl-rl-forecast', help='run name')

    # ===== Keyword Generation =====
    parser.add_argument("--description",type=str,default="",help="描述这个 dataset 的用途，例如 '新州电价数据'")
    parser.add_argument("--keyword_number",type=int,default=10,help="how many keywords to generate")
    parser.add_argument("--select_policy_by",type = str,default = "epoch",choices=["epoch", "batch"], 
                        help = "Select policy/template by epoch-level or batch-level"
    )

    args = parser.parse_args()

    if not args.tokenizer:
        args.tokenizer = args.base_model
    args.target_modules = [s.strip() for s in args.target_modules.split(',') if s.strip()]
    if args.template_ids:
        args.template_ids = [int(x) for x in args.template_ids.split(',') if x.strip()]
    else:
        args.template_ids = None
    s = args.token_budget_history_frac + args.token_budget_news_frac + args.token_budget_instr_frac
    if s > 1.0:
        args.token_budget_history_frac /= s
        args.token_budget_news_frac    /= s
        args.token_budget_instr_frac   /= s

    # 生成关键词
    
    # description = args.description.strip()
    # text = run_from_config(
    #     config_path="src/chatgpt_4o_mini_keyword_generate/config.json",
    #     kind="generate_keywords",  # 选择 A/B/C
    #     variables={
    #         "description": description if description else "null",
    #         "number": args.keyword_number
    #     },
    #     system="Be concise in your output.",
    #     temperature=0.2,
    # )

    #   # 获取输出文本
    # text = text.strip()

    # # 确保目录存在
    # out_path = Path(f"{args.keyword_path}")
    # out_path.parent.mkdir(parents=True, exist_ok=True)

    # # 写入文件
    # with open(out_path, "w", encoding="utf-8") as f:
    #     f.write(text)

    # print(f"[Keywords] Have been recorded in {out_path}")

    # 计算波动率分箱 (temp)
    # def _read(path):
    #     if path.endswith('.parquet'): return pd.read_parquet(path)
    #     return pd.read_csv(path)

    # train_df = _read(args.train_file)
    # val_df = _read(args.val_file)
    # test_df = _read(args.test_file)
    # volatility_bin  = compute_volatility_bin(train_df, time_col=args.time_col, value_col=args.value_col, window=args.history_len, bins=args.volatility_bin_tiers, dayfirst=args.dayFirst)
    # print(f"Computed volatility_bin for training set = {volatility_bin}")
    # volatility_bin_val  = compute_volatility_bin(val_df, time_col=args.time_col, value_col=args.value_col, window=args.history_len, bins=args.volatility_bin_tiers, dayfirst=args.dayFirst)
    # print(f"Computed volatility_bin for validation set = {volatility_bin_val}")
    # volatility_bin_test = compute_volatility_bin(test_df, time_col=args.time_col, value_col=args.value_col,window=args.history_len, bins=args.volatility_bin_tiers, dayfirst=args.dayFirst)
    # print(f"Computed volatility_bin for testing set = {volatility_bin_test}")


    

    main_train(args)
