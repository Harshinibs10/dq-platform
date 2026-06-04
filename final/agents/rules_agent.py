"""
RULES AGENT — Gemini version
Given column profile stats, generates AI DQ rules for each column.
"""
import json
import pandas as pd
from agents.gemini_client import generate

SYSTEM = """You are a data quality rules expert.
Given column statistics, generate specific, actionable data quality rules.
Return ONLY valid JSON:
{
  "<column_name>": [
    {"label": "short rule name", "logic": "pandas expression or description", "severity": "High|Medium|Low"}
  ]
}
Rules should be concrete and testable. Generate 2-4 rules per column based on data type and stats."""


def generate_rules(profile_stats: dict, columns_info: list) -> dict:
    """
    columns_info: list of {column, dtype, null_count, unique, samples}
    Returns: {col_name: [{label, logic, severity}]}
    """
    # Build compact stats for prompt
    compact = {}
    for col_info in columns_info:
        col = col_info["column"]
        compact[col] = {
            "dtype": col_info.get("dtype", "object"),
            "null_count": col_info.get("null_count", 0),
            "unique": col_info.get("unique", 0),
            "total": col_info.get("total", 0),
            "samples": col_info.get("samples", []),
        }
        # Add numeric stats if present
        stats = profile_stats.get(col, {})
        if "min" in stats:
            compact[col]["min"] = stats["min"]
            compact[col]["max"] = stats["max"]
            compact[col]["negative_count"] = stats.get("negative_count", 0)
        if "top_values" in stats:
            compact[col]["top_values"] = stats.get("top_values", {})

    prompt = f"""Generate data quality rules for these columns:

{json.dumps(compact, indent=2, default=str)}

For each column, generate rules based on:
- Null checks (if nulls present)
- Type validation (numeric, date, email, etc.)
- Range checks (if numeric with min/max)
- Format checks (if looks like email, phone, date, ID)
- Uniqueness (if likely primary key)
- Allowable values (if few unique values → categorical)

Return JSON with rules for all {len(columns_info)} columns."""

    try:
        raw = generate(prompt, system=SYSTEM, json_mode=True)
        return json.loads(raw)
    except Exception:
        return _fallback_rules(columns_info)


def _fallback_rules(columns_info: list) -> dict:
    rules = {}
    for col_info in columns_info:
        col = col_info["column"]
        dtype = col_info.get("dtype", "object")
        r = []
        if col_info.get("null_count", 0) > 0:
            r.append({"label": "No nulls", "logic": f"df['{col}'].notna()", "severity": "High"})
        if "int" in dtype or "float" in dtype:
            r.append({"label": "Is numeric", "logic": f"pd.to_numeric(df['{col}'], errors='coerce').notna()", "severity": "Medium"})
            r.append({"label": "Non-negative", "logic": f"df['{col}'] >= 0", "severity": "Medium"})
        else:
            r.append({"label": "Not empty string", "logic": f"df['{col}'].astype(str).str.strip() != ''", "severity": "Low"})
        if col_info.get("unique", 0) == col_info.get("total", 0):
            r.append({"label": "Unique values", "logic": f"df['{col}'].is_unique", "severity": "High"})
        rules[col] = r
    return rules
