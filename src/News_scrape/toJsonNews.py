import pandas as pd
import json
import re
from pathlib import Path


INPUT_FILE = "nasdaq_data_news_22_23.xlsx"
OUTPUT_FILE = "nasdaq_news_summary.json"


def normalize_col(col_name: str) -> str:
    """Normalize column names for easier matching."""
    if col_name is None:
        return ""
    return re.sub(r"[^a-z0-9]+", "", str(col_name).lower())


def clean_text(text) -> str:
    """Clean text fields."""
    if pd.isna(text):
        return ""
    text = str(text)

    # Fix common broken encodings / odd symbols seen in scraped text
    text = text.replace(" ?€?", "-")
    text = text.replace("??", " ")
    text = text.replace("\xa0", " ")

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def simple_summary(text: str, max_sentences: int = 3, max_chars: int = 800) -> str:
    """
    Fallback summary if no ready-made summary column exists.
    Very simple: take the first few sentences.
    """
    text = clean_text(text)
    if not text:
        return ""

    # Split on sentence endings
    sentences = re.split(r"(?<=[.!?])\s+", text)

    selected = []
    total_len = 0
    for sent in sentences:
        sent = sent.strip()
        if not sent:
            continue
        if sent in selected:
            continue
        selected.append(sent)
        total_len += len(sent)
        if len(selected) >= max_sentences or total_len >= max_chars:
            break

    summary = " ".join(selected).strip()
    if len(summary) > max_chars:
        summary = summary[:max_chars].rsplit(" ", 1)[0] + "..."
    return summary


def pick_best_content(row: pd.Series, summary_cols: list, article_col: str | None) -> str:
    """
    Prefer existing summary columns.
    If none available, summarize the full article.
    """
    candidates = []

    for col in summary_cols:
        if col in row and pd.notna(row[col]):
            value = clean_text(row[col])
            if value:
                candidates.append(value)

    # Choose the longest non-empty summary, usually more informative
    if candidates:
        return max(candidates, key=len)

    if article_col and article_col in row and pd.notna(row[article_col]):
        return simple_summary(row[article_col])

    return ""


def find_best_column(columns, candidate_names):
    """
    Find the first matching column using normalized names.
    """
    norm_map = {normalize_col(c): c for c in columns}
    for name in candidate_names:
        norm_name = normalize_col(name)
        if norm_name in norm_map:
            return norm_map[norm_name]
    return None


def process_sheet(df: pd.DataFrame) -> list:
    columns = list(df.columns)

    title_col = find_best_column(columns, [
        "Article_title", "title", "headline", "news_title"
    ])
    date_col = find_best_column(columns, [
        "Date", "date", "publish_date", "published_at", "datetime"
    ])
    url_col = find_best_column(columns, [
        "Url", "url", "link", "article_url"
    ])
    article_col = find_best_column(columns, [
        "Article", "article", "content", "body", "news_content", "text"
    ])

    possible_summary_cols = []
    for candidate in [
        "Lsa_summary", "Luhn_summary", "Textrank_summary", "Lexrank_summary",
        "summary", "abstract", "short_summary"
    ]:
        col = find_best_column(columns, [candidate])
        if col:
            possible_summary_cols.append(col)

    results = []

    for _, row in df.iterrows():
        title = clean_text(row[title_col]) if title_col and pd.notna(row[title_col]) else ""
        url = clean_text(row[url_col]) if url_col and pd.notna(row[url_col]) else ""

        # Parse and normalize date
        raw_date = row[date_col] if date_col and date_col in row else None
        parsed_date = pd.to_datetime(raw_date, errors="coerce", utc=True)
        date_str = parsed_date.strftime("%Y-%m-%dT%H:%M:%SZ") if pd.notna(parsed_date) else clean_text(raw_date)

        content = pick_best_content(row, possible_summary_cols, article_col)

        # Skip useless empty rows
        if not any([title, content, date_str, url]):
            continue

        results.append({
            "title": title,
            "content": content,
            "date": date_str,
            "url": url
        })

    return results


def main():
    input_path = Path(INPUT_FILE)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_FILE}")

    # Read all sheets
    all_sheets = pd.read_excel(input_path, sheet_name=None)

    all_results = []
    for sheet_name, df in all_sheets.items():
        if df.empty:
            continue
        sheet_results = process_sheet(df)
        all_results.extend(sheet_results)

    # Save JSON array
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    print(f"Done. Saved {len(all_results)} records to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()