import pandas as pd
import numpy as np
from datetime import timedelta
import pytz
from typing import List
from numpy.linalg import norm
from dataclasses import dataclass
import os
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from textblob import TextBlob

def _load_keywords(path: str):
    with open(path, 'r', encoding='utf-8') as f:
        return [w.strip().lower() for w in f if w.strip()]
    
def load_news(path: str, time_col: str, tz: str) -> pd.DataFrame:
    """
    Load news ONLY from a JSON file that contains an array of objects.
    - path: path to a .json file
    - time_col: name of the datetime column in the JSON objects
    - tz: target timezone string (e.g., "Australia/Sydney")
    """
    if not path.endswith(".json"):
        raise ValueError(f"Only .json files are supported, got: {path}")

    if not os.path.exists(path):
        raise FileNotFoundError(path)

    # Expect the JSON to be a list of records (array of objects)
    df = pd.read_json(path)  # if your file is JSON Lines, use: pd.read_json(path, lines=True)

    if time_col not in df.columns:
        raise KeyError(f"time_col '{time_col}' not found in JSON file.")

    # Parse as UTC-aware then convert to target timezone
    df[time_col] = pd.to_datetime(df[time_col], dayfirst=True, errors="coerce", utc=True)
    # print(df)
    df = df.dropna(subset=[time_col])

    if tz:
        df[time_col] = df[time_col].dt.tz_convert(tz)
    

    # print(df.sort_values(time_col).reset_index(drop=True))

    return df.sort_values(time_col).reset_index(drop=True)


def _align_ts_to_series_tz(ts, ref_series: pd.Series) -> pd.Timestamp:
    """将 ts 解析为 pandas.Timestamp，并对齐到 ref_series 的时区（或去掉时区）。"""
    
    ts = pd.to_datetime(ts, errors="coerce")
    if pd.isna(ts):
        return ts
    # 取新闻列的时区（可能为 None）
    tz = getattr(ref_series.dt, "tz", None)
    # print("ref_series tz:", tz)
    # print("ts.tzinfo = ",ts.tzinfo)
    if tz is not None:
        # ref 是 tz-aware
        if ts.tzinfo is None:
            ts = ts.tz_localize(tz)
        else:
            ts = ts.tz_convert(tz)
    else:
        # ref 是 tz-naive
        if ts.tzinfo:
            ts = ts.tz_convert("UTC").tz_localize(None)  # 统一转成 naive
    return ts

def get_num_news_between(news_df, time_col, target_time, window_days):
    # ✅ 先把传入的 target_time 对齐到新闻列的时区
    target_time = _align_ts_to_series_tz(target_time, news_df[time_col])

    if pd.isna(target_time):
        return 0  # 无效时间

    start = target_time - pd.Timedelta(days=window_days)
    count = news_df[(news_df[time_col] >= start) & (news_df[time_col] < target_time)].shape[0]
    return count

def get_candidates(news_df, time_col, target_time, window_days, topM):
    # ✅ 先把传入的 target_time 对齐到新闻列的时区
    # print("original target_time:", target_time)
    target_time = _align_ts_to_series_tz(target_time, news_df[time_col])

    # print("Target time aligned to news timezone:", target_time)
    if pd.isna(target_time):
        return news_df.iloc[0:0]  # 空 DataFrame：没有候选

    start = target_time - pd.Timedelta(days=window_days)
    # print("Start time for candidates:", start)
    
    cand = news_df[(news_df[time_col] >= start) & (news_df[time_col] < target_time)]
    # print(f"Found {len(cand)} candidates in the window from {start} to {target_time}")
    # 取最近 topM 条
    return cand.sort_values(time_col, ascending=False).head(topM)




def policy_by_keywords(cand: pd.DataFrame, text_col: str, keywords: List[str], K: int) -> pd.DataFrame:
    if not keywords:
        return cand.head(K)
    m = cand[text_col].fillna('').astype(str).str.lower()
    mask = m.apply(lambda x: any(kw.strip() in x for kw in keywords))
    filtered = cand[mask]
    if len(filtered) == 0:
        return cand.head(K)
    return filtered.head(K)

def select_by_sentiment_polarity_high(cand: pd.DataFrame, text_col: str, K: int) -> pd.DataFrame:
    if len(cand) == 0:
        return cand
    sentiments = cand[text_col].fillna('').astype(str).apply(lambda x: TextBlob(x).sentiment.polarity)
    order = np.argsort(-sentiments.values)
    return cand.iloc[order].head(K)

def select_by_sentiment_polarity_low(cand: pd.DataFrame, text_col: str, K: int) -> pd.DataFrame:
    if len(cand) == 0:
        return cand
    sentiments = cand[text_col].fillna('').astype(str).apply(lambda x: TextBlob(x).sentiment.polarity)
    order = np.argsort(sentiments.values)
    return cand.iloc[order].head(K)

def keyword_polarity_high_hybrid(cand: pd.DataFrame, text_col: str, keywords, K: int) -> pd.DataFrame:
    if len(cand) == 0 or K <= 0:
        return cand.head(0)

    s = cand[text_col].fillna('').astype(str)
    sentiments = s.apply(lambda x: TextBlob(x).sentiment.polarity)

    # 关键词命中
    if keywords:
        kws = [kw.strip().lower() for kw in keywords if kw and kw.strip()]
        mask = s.str.lower().apply(lambda x: any(kw in x for kw in kws))
    else:
        mask = pd.Series(False, index=cand.index)

    # 先在命中集内按情感排序
    idx_pref = sentiments[mask].sort_values(ascending=False).index.tolist()
    chosen = idx_pref[:K]

    # 不足则从非命中集里按情感排序补齐
    if len(chosen) < K:
        need = K - len(chosen)
        idx_rest = sentiments[~mask].sort_values(ascending=False).index.tolist()
        chosen += idx_rest[:need]

    return cand.loc[chosen]

def keyword_polarity_low_hybrid(cand: pd.DataFrame, text_col: str, keywords, K: int) -> pd.DataFrame:
    if len(cand) == 0 or K <= 0:
        return cand.head(0)

    s = cand[text_col].fillna('').astype(str)
    sentiments = s.apply(lambda x: TextBlob(x).sentiment.polarity)

    # 关键词命中
    if keywords:
        kws = [kw.strip().lower() for kw in keywords if kw and kw.strip()]
        mask = s.str.lower().apply(lambda x: any(kw in x for kw in kws))
    else:
        mask = pd.Series(False, index=cand.index)

    # 先在命中集内按情感排序
    idx_pref = sentiments[mask].sort_values(ascending=True).index.tolist()
    chosen = idx_pref[:K]

    # 不足则从非命中集里按情感排序补齐
    if len(chosen) < K:
        need = K - len(chosen)
        idx_rest = sentiments[~mask].sort_values(ascending=True).index.tolist()
        chosen += idx_rest[:need]

    return cand.loc[chosen]

def select_sum_v0(cand, text_col,K):
    m = cand[text_col].fillna('').astype(str).str.lower()
    filtered = cand[m]
    return filtered.head(K)

def select_no_sum(cand, text_col, K):
    m = cand[text_col].fillna('').astype(str).str.lower()

    filtered = cand[m]
    return filtered.head(K)

def select_news(cand: pd.DataFrame, policy: str, text_col: str,
                policy_kw: List[str], K: int) -> pd.DataFrame:
    # print(cand)
    selected = cand
    if policy == 'keywords':
        selected = policy_by_keywords(cand, text_col, policy_kw, K) #这里把命中关键词的都选上
    elif policy == 'polarity_high':
        selected = select_by_sentiment_polarity_high(cand, text_col, K)#选个
    elif policy == 'polarity_low':
        selected = select_by_sentiment_polarity_low(cand, text_col, K)#选个
    elif policy == 'keyword_polarity_high_hybrid':
        selected = keyword_polarity_high_hybrid(cand, text_col, policy_kw, K)#先按keyword筛选，然后选个
    elif policy == 'keyword_polarity_low_hybrid':
        selected = keyword_polarity_low_hybrid(cand, text_col, policy_kw, 1)#先按keyword筛选，然后选个
    elif policy == "all":
        selected = cand
    elif policy == "sum_v0":
        selected = cand
    elif policy == "no_sum":
        selected = cand
    else:
        raise ValueError("Unknown news select policy")
    # Return both selected news and average rating for use in base model training.
    avg_rate = selected['rate'].mean() if not selected.empty else 0.0
    return selected, avg_rate

def lead3(text: str, max_sentences: int = 3) -> str:
    seps = ['。', '.', '；', ';', '！', '!', '？', '?', '\n']
    tmp = text
    for s in seps:
        tmp = tmp.replace(s, '.')
    parts = [p.strip() for p in tmp.split('.') if p.strip()]
    return ' '.join(parts[:max_sentences])
