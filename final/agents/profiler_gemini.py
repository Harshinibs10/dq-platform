"""
PROFILER AGENT — Gemini version
Profiles a DataFrame and returns column stats + issue list.
"""
import json
import pandas as pd
from agents.gemini_client import generate

SYSTEM = """You are a data quality profiler. 
Given column statistics, return a JSON with:
{
  "summary": "one-sentence summary",
  "issue_count": N,
  "column_issues": [
    {"column": "col", "issue_type": "null|duplicate|outlier|type|format|range", "severity": "High|Medium|Low", "detail": "..."}
  ],
  "recommended_checks": ["description of check 1", ...]
}
Return ONLY valid JSON."""

def profile_dataframe(df: pd.DataFrame, table_name: str = "dataset") -> dict:
    stats = _compute_stats(df)
    prompt = f"""Profile this dataset '{table_name}' and identify data quality issues:

{json.dumps(stats, indent=2, default=str)}

Identify nulls, outliers, duplicates, format inconsistencies, type mismatches, and range violations.
Return the JSON profile."""

    try:
        raw = generate(prompt, system=SYSTEM, json_mode=True)
        result = json.loads(raw)
    except Exception:
        result = {"summary": "Profile complete", "issue_count": 0, "column_issues": [], "recommended_checks": []}

    result["_stats"] = stats
    result["table_name"] = table_name
    result["row_count"] = len(df)
    result["col_count"] = len(df.columns)
    return result


def _compute_stats(df: pd.DataFrame) -> dict:
    out = {}
    for col in df.columns:
        s = df[col]
        info = {
            "dtype": str(s.dtype),
            "null_count": int(s.isna().sum()),
            "null_pct": round(float(s.isna().mean()) * 100, 2),
            "unique_count": int(s.nunique()),
            "total": len(s),
            "samples": s.dropna().astype(str).head(3).tolist(),
        }
        if pd.api.types.is_numeric_dtype(s) and s.notna().any():
            info.update({
                "min": float(s.min()), "max": float(s.max()),
                "mean": round(float(s.mean()), 4),
                "negative_count": int((s < 0).sum()),
                "zero_count": int((s == 0).sum()),
            })
        else:
            top = s.value_counts().head(3).to_dict()
            info["top_values"] = {str(k): int(v) for k, v in top.items()}
        out[col] = info
    return out
