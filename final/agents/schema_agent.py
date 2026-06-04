"""
SCHEMA ANALYZER AGENT
─────────────────────
Analyzes multiple uploaded tables/CSVs and returns:
  - Primary key candidates per table
  - Foreign key relationships between tables
  - Column types & descriptions
  - Join recommendations
Uses Gemini Flash (free tier).
"""
import json, pandas as pd
from agents.gemini_client import generate

SYSTEM = """You are a database schema expert. 
Analyze the provided table metadata and return a JSON object with:
{
  "tables": {
    "<table_name>": {
      "primary_key_candidates": ["col1"],
      "description": "brief description",
      "columns": {
        "<col_name>": {"role": "pk|fk|measure|dimension|id|date|text", "description": "..."}
      }
    }
  },
  "relationships": [
    {"from_table": "t1", "from_col": "c1", "to_table": "t2", "to_col": "c2", "confidence": 0.9, "type": "many-to-one"}
  ],
  "recommended_joins": [
    {"description": "...", "sql": "SELECT ... FROM t1 JOIN t2 ON ..."}
  ]
}
Return ONLY valid JSON. No markdown fences."""


def analyze_schema(tables: dict[str, pd.DataFrame]) -> dict:
    """
    tables: {table_name: dataframe}
    Returns schema analysis dict.
    """
    meta = {}
    for name, df in tables.items():
        col_info = {}
        for col in df.columns:
            s = df[col]
            col_info[col] = {
                "dtype": str(s.dtype),
                "null_pct": round(float(s.isna().mean()) * 100, 1),
                "unique_count": int(s.nunique()),
                "total_rows": len(s),
                "sample_values": s.dropna().astype(str).head(5).tolist(),
                "all_unique": bool(s.nunique() == len(s.dropna())),
            }
        meta[name] = {
            "row_count": len(df),
            "col_count": len(df.columns),
            "columns": col_info,
        }

    prompt = f"""Analyze these {len(tables)} tables and identify schema structure, primary keys, foreign keys, and relationships.

Table metadata:
{json.dumps(meta, indent=2)}

Focus on:
1. Columns with all-unique values → likely primary keys
2. Column names ending in _id, _key, _code that appear in multiple tables → likely foreign keys
3. Numeric columns that look like IDs (integers, no decimals, sequential) → key candidates
4. Date/timestamp columns
5. Cross-table column name matches → relationships

Return the JSON schema analysis."""

    try:
        raw = generate(prompt, system=SYSTEM, json_mode=True)
        return json.loads(raw)
    except Exception:
        return _basic_schema(tables)


def _basic_schema(tables: dict) -> dict:
    """Fallback schema without LLM."""
    schema = {"tables": {}, "relationships": [], "recommended_joins": []}
    all_cols = {}  # col_name -> [table_names]
    for name, df in tables.items():
        pk_candidates = []
        columns = {}
        for col in df.columns:
            s = df[col]
            is_unique = s.nunique() == len(s.dropna())
            role = "measure"
            if is_unique and (col.lower().endswith("_id") or col.lower() == "id"):
                role = "pk"
                pk_candidates.append(col)
            elif col.lower().endswith("_id") or col.lower().endswith("_key"):
                role = "fk"
            elif "date" in col.lower() or "time" in col.lower():
                role = "date"
            elif str(s.dtype) == "object":
                role = "dimension"
            columns[col] = {"role": role, "description": col.replace("_", " ").title()}
            all_cols.setdefault(col.lower(), []).append(name)
        schema["tables"][name] = {
            "primary_key_candidates": pk_candidates,
            "description": f"Table with {len(df)} rows and {len(df.columns)} columns",
            "columns": columns,
        }
    # Find relationships from shared column names
    for col_lower, tbl_list in all_cols.items():
        if len(tbl_list) > 1:
            for i in range(len(tbl_list) - 1):
                schema["relationships"].append({
                    "from_table": tbl_list[i],
                    "from_col": col_lower,
                    "to_table": tbl_list[i + 1],
                    "to_col": col_lower,
                    "confidence": 0.7,
                    "type": "unknown",
                })
    return schema
