"""
DQ Platform — FastAPI backend (Gemini edition)
Run: uvicorn ui.app:app --reload --port 8000
"""
import os, sys, json, uuid, threading, traceback, io, time
from pathlib import Path
from datetime import datetime
from typing import Optional, List

sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
load_dotenv()

import pandas as pd
import numpy as np

app = FastAPI(title="DQ Platform", version="4.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

UPLOAD_DIR  = Path("data/uploads")
REPORTS_DIR = Path("reports")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# ── In-memory state ────────────────────────────────────────────────────────
SESSIONS: dict = {}   # session_id → {tables, profile, schema}
RUNS:     dict = {}   # run_id → {status, log_queue, results, ...}

# ─────────────────────────────────────────────────────────────────────────────
# SERVE FRONTEND
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    html_path = Path(__file__).parent / "index.html"
    return HTMLResponse(html_path.read_text(encoding="utf-8"))

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1: UPLOAD FILES
# ─────────────────────────────────────────────────────────────────────────────
@app.post("/api/upload")
async def upload_files(files: List[UploadFile] = File(...)):
    """Accept 1+ CSV/Excel files, store them, return session_id + table previews."""
    session_id = str(uuid.uuid4())[:8]
    session_dir = UPLOAD_DIR / session_id
    raw_dir = session_dir / "_raw"       # ← raw uploads go here, NOT in session root
    raw_dir.mkdir(parents=True, exist_ok=True)

    tables_meta = {}
    for f in files:
        ext  = Path(f.filename).suffix.lower()
        name = Path(f.filename).stem
        dest = raw_dir / f.filename      # ← save raw file in _raw/
        dest.write_bytes(await f.read())

        if ext in (".xlsx", ".xls"):
            xl = pd.ExcelFile(dest)
            for sheet in xl.sheet_names:
                df = xl.parse(sheet)
                tname = f"{name}__{sheet}" if len(xl.sheet_names) > 1 else name
                _save_parquet(df, session_id, tname)
                tables_meta[tname] = _table_meta(df, tname)
        else:
            df = pd.read_csv(dest)
            _save_parquet(df, session_id, name)
            tables_meta[name] = _table_meta(df, name)

    SESSIONS[session_id] = {"tables_meta": tables_meta, "schema": None, "profile": {}}
    return {"session_id": session_id, "tables": tables_meta}

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1b: CONNECT TO SQL SERVER (SSMS)
# ─────────────────────────────────────────────────────────────────────────────
class SSMSConnectRequest(BaseModel):
    server: str
    database: str
    username: Optional[str] = None
    password: Optional[str] = None
    table_name: str
    use_windows_auth: bool = False
    query: Optional[str] = None

@app.post("/api/connect-ssms")
async def connect_ssms(req: SSMSConnectRequest):
    """Connect to SQL Server and load a table into the session."""
    try:
        import pyodbc
    except ImportError:
        raise HTTPException(400, "pyodbc not installed. Run: pip install pyodbc")

    try:
        if req.use_windows_auth:
            conn_str = (
                f"DRIVER={{ODBC Driver 17 for SQL Server}};"
                f"SERVER={req.server};DATABASE={req.database};"
                f"Trusted_Connection=yes;"
            )
        else:
            conn_str = (
                f"DRIVER={{ODBC Driver 17 for SQL Server}};"
                f"SERVER={req.server};DATABASE={req.database};"
                f"UID={req.username};PWD={req.password};"
            )

        conn = pyodbc.connect(conn_str, timeout=15)
        sql = req.query if req.query else f"SELECT TOP 50000 * FROM [{req.table_name}]"
        df = pd.read_sql(sql, conn)
        conn.close()

        session_id = str(uuid.uuid4())[:8]
        table_name = req.table_name.replace(" ", "_")
        _save_parquet(df, session_id, table_name)

        tables_meta = {table_name: _table_meta(df, table_name)}
        SESSIONS[session_id] = {"tables_meta": tables_meta, "schema": None, "profile": {}}
        return {"session_id": session_id, "tables": tables_meta}

    except Exception as e:
        raise HTTPException(500, f"SQL Server connection failed: {str(e)}")


@app.post("/api/schema/{session_id}")
async def analyze_schema(session_id: str):
    if session_id not in SESSIONS:
        raise HTTPException(404, "Session not found")
    tables = _load_all_tables(session_id)
    if not tables:
        raise HTTPException(400, "No tables found")

    from agents.schema_agent import analyze_schema as _analyze
    schema = _analyze(tables)
    SESSIONS[session_id]["schema"] = schema
    return schema

# ─────────────────────────────────────────────────────────────────────────────
# STEP 3: LOAD COLUMNS + GENERATE AI RULES
# ─────────────────────────────────────────────────────────────────────────────
class LoadColumnsRequest(BaseModel):
    session_id: str
    table_names: List[str]  # which tables to include

@app.post("/api/load-columns")
async def load_columns(req: LoadColumnsRequest):
    if req.session_id not in SESSIONS:
        raise HTTPException(404, "Session not found")

    tables = _load_all_tables(req.session_id)
    # Merge selected tables into one DF (union columns)
    selected = {k: v for k, v in tables.items() if k in req.table_names}
    if not selected:
        selected = tables  # fallback: use all

    # ── Build per-table column profiles BEFORE merging ──────────────────────
    # This preserves which column belongs to which table for the frontend.
    col_profiles = []
    raw_stats = {}
    seen_cols = set()

    for tname, tdf in selected.items():
        for col in tdf.columns:
            # De-duplicate columns that appear in multiple tables
            col_key = f"{tname}.{col}" if col in seen_cols else col
            seen_cols.add(col)

            s = tdf[col]
            col_profiles.append({
                "column": col,
                "source_table": tname,          # ← key fix: tag each column with its table
                "dtype": str(s.dtype),
                "null_count": int(s.isna().sum()),
                "unique": int(s.nunique()),
                "total": len(tdf),
                "samples": s.dropna().astype(str).head(3).tolist(),
            })

            stat = {"null_count": int(s.isna().sum()), "unique_count": int(s.nunique())}
            if pd.api.types.is_numeric_dtype(s) and s.notna().any():
                stat.update({"min": float(s.min()), "max": float(s.max()),
                             "negative_count": int((s < 0).sum())})
            else:
                top = s.value_counts().head(5).to_dict()
                stat["top_values"] = {str(k): int(v) for k, v in top.items()}
            raw_stats[col] = stat

    # ── Build merged DF for the pipeline run ────────────────────────────────
    if len(selected) == 1:
        df = list(selected.values())[0]
        table_name = list(selected.keys())[0]
    else:
        dfs = []
        for tname, tdf in selected.items():
            tmp = tdf.copy()
            tmp["__source_table__"] = tname
            dfs.append(tmp)
        df = pd.concat(dfs, ignore_index=True, sort=False)
        table_name = "+".join(selected.keys())

    # ── Generate AI rules per column ─────────────────────────────────────────
    from agents.rules_agent import generate_rules
    ai_rules = generate_rules(raw_stats, col_profiles)

    # ── Tag each rule with its source table (for grouping in results) ────────
    for cp in col_profiles:
        col = cp["column"]
        tbl = cp["source_table"]
        for rule in ai_rules.get(col, []):
            rule.setdefault("source_table", tbl)

    # Store merged df for this session's run
    _save_parquet(df, req.session_id, "__merged__")

    profile = {
        "table_name": table_name,
        "row_count": len(df),
        "col_count": len(df.columns),
        "columns": col_profiles,
        "tables": list(selected.keys()),          # ← list of table names for UI
        "table_row_counts": {k: len(v) for k, v in selected.items()},
    }
    SESSIONS[req.session_id]["profile"] = profile
    SESSIONS[req.session_id]["raw_stats"] = raw_stats

    return {
        "session_id": req.session_id,
        "profile": profile,
        "ai_rules": ai_rules,
    }

# ─────────────────────────────────────────────────────────────────────────────
# STEP 4: RUN PIPELINE
# ─────────────────────────────────────────────────────────────────────────────
class RunRequest(BaseModel):
    session_id: str
    column_rules: dict  # {col: [{label, logic, severity, source}]}

@app.post("/api/run")
async def run_pipeline(req: RunRequest):
    if req.session_id not in SESSIONS:
        raise HTTPException(404, "Session not found")

    run_id = str(uuid.uuid4())[:8]
    RUNS[run_id] = {
        "status": "queued",
        "log": [],
        "results": [],
        "fixes_applied": 0,
        "started": datetime.utcnow().isoformat(),
        "column_rules": req.column_rules,          # ← store rules for download
        "session_id": req.session_id,              # ← store session for table info
    }

    t = threading.Thread(
        target=_run_pipeline_thread,
        args=(run_id, req.session_id, req.column_rules),
        daemon=True,
    )
    t.start()
    return {"run_id": run_id}

@app.get("/api/run/{run_id}/stream")
async def stream_logs(run_id: str):
    """SSE stream of pipeline logs."""
    if run_id not in RUNS:
        raise HTTPException(404)

    async def event_gen():
        seen = 0
        for _ in range(300):  # max 300 polls = ~150s
            run = RUNS.get(run_id, {})
            logs = run.get("log", [])
            while seen < len(logs):
                entry = logs[seen]
                yield f"data: {json.dumps(entry)}\n\n"
                seen += 1
            if run.get("status") in ("done", "error"):
                yield f"data: {json.dumps({'type':'done','status':run['status']})}\n\n"
                return
            await __import__("asyncio").sleep(0.5)

    return StreamingResponse(event_gen(), media_type="text/event-stream")

@app.get("/api/run/{run_id}/status")
async def run_status(run_id: str):
    if run_id not in RUNS:
        raise HTTPException(404)
    run = RUNS[run_id]
    return {"status": "COMPLETED" if run["status"] == "done" else run["status"].upper()}

@app.get("/api/run/{run_id}/results")
async def run_results(run_id: str):
    if run_id not in RUNS:
        raise HTTPException(404)
    return RUNS[run_id].get("results", [])

@app.get("/api/run/{run_id}/download")
async def download_cleaned(run_id: str):
    path = REPORTS_DIR / run_id / "cleaned_data.csv"
    if not path.exists():
        raise HTTPException(404, "File not ready")

    run = RUNS.get(run_id, {})
    column_rules: dict = run.get("column_rules", {})
    session_id = run.get("session_id")
    session_profile = SESSIONS.get(session_id, {}).get("profile", {})
    default_table_name = session_profile.get("table_name", "data")

    # Load the cleaned CSV
    df = pd.read_csv(path)

    # Determine per-row, per-column pass/fail by re-running rule checks
    # We re-use the same logic as ValidatorAgent._execute_rule but return a boolean mask per row.
    def _row_mask_for_rule(s: pd.Series, combined: str, original_logic: str, n: int) -> "pd.Series":
        """Return a boolean Series: True = row PASSES this rule, False = row FAILS.
        Mirrors _execute_rule in validator_gemini.py — keep in sync."""
        import re as _re
        all_pass = pd.Series([True] * n, index=s.index)

        # 1. PATTERN / REGEX — before cross-column ("match" clash)
        pat_m = _re.search(r"must match pattern[:\s]+(.+?)(?:\s*\(case-insensitive\))?\s*$", combined)
        if pat_m:
            pattern = pat_m.group(1).strip()
            flags = _re.IGNORECASE if "case-insensitive" in combined else 0
            try:
                bad = ~s.astype(str).str.match(pattern, flags=flags) & s.notna()
                return ~bad
            except _re.error:
                return all_pass

        # 2. EMPTY CHECK / NULL
        if any(k in combined for k in ["empty or null", "null", "not null", "notna",
                                        "no null", "missing", "non-null", "non null"]):
            bad = s.isna()
            if "empty or null" in combined:
                bad = bad | (s.astype(str).str.strip() == "")
            if any(k in combined for k in ["not", "no null", "non", "must not"]):
                return ~bad
            else:
                return ~bad

        # 3. NO WHITESPACE — before empty-string check
        if "whitespace" in combined:
            if "anywhere" in combined:
                bad = s.astype(str).str.contains(r"\s", regex=True) & s.notna()
            elif "leading" in combined and "trailing" in combined:
                bad = (s.astype(str).str.match(r"^\s") | s.astype(str).str.match(r".*\s$")) & s.notna()
            elif "leading" in combined:
                bad = s.astype(str).str.match(r"^\s") & s.notna()
            elif "trailing" in combined:
                bad = s.astype(str).str.match(r".*\s$") & s.notna()
            else:
                bad = (s.astype(str).str.match(r"^\s") | s.astype(str).str.match(r".*\s$")) & s.notna()
            return ~bad

        # 4. EMPTY STRING
        if any(k in combined for k in ["empty string", "not empty", "blank"]):
            bad = s.astype(str).str.strip() == ""
            return ~bad

        # 5. UNIQUE / DUPLICATE
        if any(k in combined for k in ["unique", "duplicate", "distinct", "no dup"]):
            return ~s.duplicated(keep=False)

        # 6. IS NUMBER — before positive/allowed_values ("must be" clash)
        if any(k in combined for k in ["integer only", "int or decimal", "decimal/float",
                                        "digits-only string", "must be a numeric",
                                        "is numeric", "numeric type", "numeric value", "valid number"]):
            if "digits-only string" in combined:
                bad = ~s.astype(str).str.match(r"^\d+$") & s.notna()
            elif "integer only" in combined:
                num = pd.to_numeric(s, errors="coerce")
                bad = (num.isna() | (num % 1 != 0)) & s.notna()
            elif "decimal/float" in combined:
                bad = pd.to_numeric(s, errors="coerce").isna() & s.notna()
            else:
                num = pd.to_numeric(s, errors="coerce")
                bad = num.isna() & s.notna()
                if "positive only" in combined:
                    bad = bad | ((num <= 0) & s.notna())
            return ~bad

        # 7. POSITIVE / NON-NEGATIVE — before allowed_values
        if any(k in combined for k in ["positive number greater than 0",
                                        "must be a positive", "greater than 0"]):
            num = pd.to_numeric(s, errors="coerce")
            bad = ((num <= 0) | num.isna()) & s.notna()
            return ~bad

        if any(k in combined for k in ["greater than or equal to 0", "non-negative",
                                        "nonnegative", "not negative", ">= 0", "\u2265 0"]):
            bad = (pd.to_numeric(s, errors="coerce") < 0) & s.notna()
            return ~bad

        # 8. EMAIL
        if any(k in combined for k in ["valid email address", "email", "e-mail"]):
            general_pattern = r"^[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}$"
            valid_mask = s.astype(str).str.match(general_pattern)
            domain_src = _re.search(r"allowed domains?[:\s]+([a-zA-Z0-9.,\s\-]+)", combined)
            if not domain_src:
                domain_src = _re.search(r"(?:valid email|email).*?\(([a-zA-Z0-9.\-,\s]+)\)", combined)
            if domain_src:
                allowed_domains = [d.strip().lower() for d in _re.split(r"[,;]", domain_src.group(1)) if d.strip()]
                if allowed_domains:
                    dom_re = "|".join(_re.escape(f"@{d}") + r"$" for d in allowed_domains)
                    domain_ok = s.astype(str).str.lower().str.contains(dom_re, regex=True)
                    bad = (~(valid_mask & domain_ok)) & s.notna()
                    return ~bad
            bad = ~valid_mask & s.notna()
            return ~bad

        # 9. PHONE
        if any(k in combined for k in ["valid phone number", "phone", "mobile", "contact number"]):
            dlen = _re.search(r"(\d+)\s*digits?\s*(?:only|exact|exactly)?|"
                              r"(?:only|exactly)\s*(\d+)\s*digits?", combined)
            if dlen:
                req = int(dlen.group(1) or dlen.group(2))
                norm = s.astype(str).str.replace(r"[\s\-\+\(\)]", "", regex=True)
                bad = ~norm.str.match(rf"^\d{{{req}}}$") & s.notna()
            elif "international" in combined or "intl" in combined:
                norm = s.astype(str).str.replace(r"[\s\-\(\)]", "", regex=True)
                bad = ~norm.str.match(r"^\+?\d{7,14}$") & s.notna()
            elif _re.search(r"\bus\b", combined) or "xxx" in combined:
                norm = s.astype(str).str.replace(r"[\s\-\+\(\)]", "", regex=True)
                bad = ~norm.str.match(r"^\d{10}$") & s.notna()
            else:
                norm = s.astype(str).str.replace(r"[\s\-\+\(\)]", "", regex=True)
                bad = ~norm.str.match(r"^\d{7,15}$") & s.notna()
            return ~bad

        # 10. DATE
        if any(k in combined for k in ["valid date", "date", "datetime", "timestamp", "date format"]):
            try:
                parsed = pd.to_datetime(s, errors="coerce")
                bad = parsed.isna() & s.notna()
                not_before = _re.search(r"not before\s+(\d{4}-\d{2}-\d{2}|\d{2}/\d{2}/\d{4})", combined)
                not_after  = _re.search(r"not after\s+(\d{4}-\d{2}-\d{2}|\d{2}/\d{2}/\d{4})", combined)
                if not_before:
                    bad = bad | ((parsed < pd.to_datetime(not_before.group(1))) & parsed.notna())
                if not_after:
                    bad = bad | ((parsed > pd.to_datetime(not_after.group(1))) & parsed.notna())
                return ~bad
            except Exception:
                return all_pass

        # 11. LENGTH
        if "length in" in combined:
            lmin_m = _re.search(r"between\s+(\d+)", combined)
            lmax_m = _re.search(r"and\s+(\d+)", combined)
            lmin = int(lmin_m.group(1)) if lmin_m else 0
            actual = s.astype(str).str.split().str.len() if "words" in combined else s.astype(str).str.len()
            bad = actual < lmin
            if lmax_m:
                lmax = int(lmax_m.group(1))
                bad = bad | (actual > lmax)
            bad = bad & s.notna()
            return ~bad

        len_eq_m = _re.search(r"length\s*(?:==|=|is)\s*(\d+)", combined)
        if len_eq_m:
            elen = int(len_eq_m.group(1))
            return ~((s.astype(str).str.len() != elen) & s.notna())

        # 12. RANGE
        if "between" in combined:
            nums = _re.findall(r"-?[0-9]+(?:\.[0-9]+)?", combined)
            if len(nums) >= 2:
                lo, hi = float(nums[0]), float(nums[-1])
                num = pd.to_numeric(s, errors="coerce")
                strict = "strictly" in combined
                bad = ((num <= lo if strict else num < lo) |
                       (num >= hi if strict else num > hi)) & s.notna()
                return ~bad

        rmin = _re.search(r"(?:>=|\u2265|min|minimum)\s*([0-9]+(?:\.[0-9]+)?)", combined)
        rmax = _re.search(r"(?:<=|\u2264|max|maximum)\s*([0-9]+(?:\.[0-9]+)?)", combined)
        if rmin or rmax:
            num = pd.to_numeric(s, errors="coerce")
            bad = pd.Series([False] * n, index=s.index)
            if rmin: bad = bad | ((num < float(rmin.group(1))) & s.notna())
            if rmax: bad = bad | ((num > float(rmax.group(1))) & s.notna())
            return ~bad

        # 13. ALLOWED VALUES — last to avoid "must be" false positives
        av_m = _re.search(
            r"(?:one of\s*[:\[]|in\s*\[|allowed values?\s*[:\[]?|valid values?\s*[:\[]?|allowed\s*:)"
            r"\s*\[?\s*([\w,\s\'\".@\-]+?)(?:\]|$)",
            combined
        )
        if av_m:
            raw = av_m.group(1)
            vals = {v.strip().strip("\'\"").lower() for v in _re.split(r"[,;]| or ", raw) if v.strip()}
            if vals:
                bad = ~s.astype(str).str.lower().isin(vals) & s.notna()
                return ~bad

        # DEFAULT: all pass
        return all_pass

    # Build per-column fail mask: col → boolean Series (True = at least one rule failed for that row)
    col_fail_mask: dict = {}   # col → pd.Series of bool (True = FAIL)
    checked_cols = set()
    for col, rules in column_rules.items():
        if col not in df.columns:
            continue
        checked_cols.add(col)
        s = df[col]
        n = len(df)
        row_any_fail = pd.Series([False] * n, index=df.index)
        for rule in rules:
            label = rule.get("label", "").lower()
            logic = rule.get("logic", "").lower()
            combined = label + " " + logic
            try:
                row_pass = _row_mask_for_rule(s, combined, rule.get("logic", ""), n)
                row_any_fail = row_any_fail | ~row_pass
            except Exception:
                pass
        col_fail_mask[col] = row_any_fail

    # Detect if multiple tables — split by __source_table__ column if present
    has_source_col = "__source_table__" in df.columns

    if has_source_col:
        tables_in_df = df["__source_table__"].unique().tolist()
    else:
        tables_in_df = [default_table_name]

    # Build per-table column mapping from session profile
    # col_profiles has {column, source_table} — use it to know which cols belong to which table
    col_profiles = session_profile.get("columns", [])
    table_own_cols = {}  # tbl -> [col, col, ...]
    for cp in col_profiles:
        tbl = cp.get("source_table") or default_table_name
        table_own_cols.setdefault(tbl, []).append(cp["column"])

    # Build one sheet per table — only that table's own columns
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        for tbl in tables_in_df:
            if has_source_col:
                tbl_mask = df["__source_table__"] == tbl
                tbl_df = df[tbl_mask].reset_index(drop=True)
            else:
                tbl_df = df.reset_index(drop=True)

            # Only columns that actually belong to this table
            own_cols = table_own_cols.get(tbl) or [c for c in tbl_df.columns if c != "__source_table__"]

            frames = {}
            for col in own_cols:
                if col not in tbl_df.columns:
                    continue
                frames[col] = tbl_df[col].values
                if col in col_fail_mask:
                    if has_source_col:
                        fail_vals = col_fail_mask[col][tbl_mask].reset_index(drop=True).values
                    else:
                        fail_vals = col_fail_mask[col].values
                    # Build analysis value: "PASS" or "FAIL - <reason>"
                    import re as _re2
                    col_series = tbl_df[col].reset_index(drop=True)
                    rules_for_col = column_rules.get(col, [])
                    analysis_vals = []
                    for row_idx, is_fail in enumerate(fail_vals):
                        if not is_fail:
                            analysis_vals.append("PASS")
                            continue
                        cell_val = col_series.iloc[row_idx]
                        failed_reasons = []
                        for rule in rules_for_col:
                            label = rule.get("label", "").lower()
                            logic = rule.get("logic", "").lower()
                            combined = label + " " + logic
                            try:
                                row_pass = _row_mask_for_rule(col_series, combined, rule.get("logic", ""), len(tbl_df))
                                if not row_pass.iloc[row_idx]:
                                    if pd.isna(cell_val) or str(cell_val).strip() in ("", "nan"):
                                        failed_reasons.append("Null value")
                                    elif any(k in combined for k in ["unique", "duplicate", "distinct"]):
                                        failed_reasons.append("Duplicate value")
                                    elif any(k in combined for k in ["email", "e-mail"]):
                                        dom_src = _re2.search(r"(?:allowed domains?[:\s]+|valid email.*?\()([a-zA-Z0-9.,\s\-]+)", combined)
                                        if dom_src:
                                            failed_reasons.append(f"Invalid email domain (expected: {dom_src.group(1).strip()})")
                                        else:
                                            failed_reasons.append("Invalid email format")
                                    elif any(k in combined for k in ["phone", "mobile", "contact number"]):
                                        dlen = _re2.search(r"(\d+)\s*digits?", combined)
                                        if dlen:
                                            failed_reasons.append(f"Invalid phone (expected {dlen.group(1)} digits)")
                                        else:
                                            failed_reasons.append("Invalid phone number")
                                    elif "between" in combined:
                                        nums = _re2.findall(r"-?[0-9]+(?:\.[0-9]+)?", combined)
                                        if len(nums) >= 2:
                                            failed_reasons.append(f"Value out of range ({nums[0]}-{nums[-1]})")
                                        else:
                                            failed_reasons.append("Value out of allowed range")
                                    elif any(k in combined for k in [">=", "<=", "min", "max", "greater than", "less than"]):
                                        rmin = _re2.search(r"(?:>=|min|minimum)\s*([0-9]+(?:\.[0-9]+)?)", combined)
                                        rmax = _re2.search(r"(?:<=|max|maximum)\s*([0-9]+(?:\.[0-9]+)?)", combined)
                                        if rmin and rmax:
                                            failed_reasons.append(f"Value out of range (min {rmin.group(1)}, max {rmax.group(1)})")
                                        elif rmin:
                                            failed_reasons.append(f"Value below minimum ({rmin.group(1)})")
                                        elif rmax:
                                            failed_reasons.append(f"Value exceeds maximum ({rmax.group(1)})")
                                        else:
                                            failed_reasons.append("Value out of allowed range")
                                    elif any(k in combined for k in ["one of", "allowed values", "valid values"]):
                                        av = _re2.search(r"(?:one of|allowed values?|valid values?)[:\s\[]+([^\]]+)", combined)
                                        if av:
                                            failed_reasons.append(f"Invalid value (allowed: {av.group(1).strip()})")
                                        else:
                                            failed_reasons.append("Value not in allowed list")
                                    elif any(k in combined for k in ["integer only", "numeric", "valid number", "decimal"]):
                                        failed_reasons.append("Non-numeric value")
                                    elif any(k in combined for k in ["positive", "greater than 0", "non-negative"]):
                                        failed_reasons.append(f"Value must be positive (got: {cell_val})")
                                    elif any(k in combined for k in ["date", "datetime", "timestamp"]):
                                        failed_reasons.append("Invalid date format")
                                    elif "length" in combined:
                                        actual_len = len(str(cell_val))
                                        failed_reasons.append(f"Invalid length (got {actual_len} chars)")
                                    elif "pattern" in combined:
                                        pat = _re2.search(r"must match pattern[:\s]+(.+?)(?:\s*\(|$)", combined)
                                        if pat:
                                            failed_reasons.append(f"Does not match pattern: {pat.group(1).strip()}")
                                        else:
                                            failed_reasons.append("Pattern mismatch")
                                    elif "whitespace" in combined:
                                        failed_reasons.append("Contains unwanted whitespace")
                                    else:
                                        failed_reasons.append(rule.get("label", "Rule violation"))
                            except Exception:
                                pass
                        reason_str = "; ".join(failed_reasons) if failed_reasons else "Failed validation"
                        analysis_vals.append(f"FAIL - {reason_str}")
                    frames[f"{col}_analysis"] = analysis_vals

                elif col in checked_cols:
                    frames[f"{col}_analysis"] = ["PASS"] * len(tbl_df)

            # Build summary column: PASS only if ALL _analysis columns start with "PASS", else FAIL
            analysis_keys = [k for k in frames if k.endswith("_analysis")]
            if analysis_keys:
                row_count = len(tbl_df)
                summary = []
                for i in range(row_count):
                    all_pass = all(str(frames[ak][i]).split(" - ")[0].strip().upper() == "PASS" for ak in analysis_keys)
                    summary.append("PASS" if all_pass else "FAIL")
                frames["summary"] = summary

            sheet_name = tbl[:31]
            pd.DataFrame(frames).to_excel(writer, sheet_name=sheet_name, index=False)

    buf.seek(0)
    return StreamingResponse(
        iter([buf.read()]),
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": "attachment; filename=cleaned_data_with_analysis.xlsx"}
    )

# ─────────────────────────────────────────────────────────────────────────────
# DASHBOARD / HISTORY APIs
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/api/dashboard-stats")
async def dashboard_stats():
    runs = list(RUNS.values())
    done = [r for r in runs if r["status"] == "done"]
    total_rules = sum(len(r.get("results", [])) for r in done)
    avg_risk = 0
    if done:
        pass_rates = []
        for r in done:
            res = r.get("results", [])
            if res:
                passed = sum(1 for x in res if x["status"] == "PASS")
                pass_rates.append(round((1 - passed / len(res)) * 100))
        avg_risk = round(sum(pass_rates) / len(pass_rates)) if pass_rates else 0

    recent = []
    for rid, r in list(RUNS.items())[-10:]:
        res = r.get("results", [])
        passed = sum(1 for x in res if x["status"] == "PASS")
        failed = sum(1 for x in res if x["status"] != "PASS")
        fail_pct = (failed / len(res) * 100) if res else 0
        risk = "LOW" if fail_pct < 10 else ("MEDIUM" if fail_pct < 30 else ("HIGH" if fail_pct < 60 else "CRITICAL"))
        recent.append({
            "run_id": rid,
            "started_at": r.get("started", ""),
            "total_checks": len(res),
            "passed": passed,
            "failed": failed,
            "risk_level": risk,
            "status": "COMPLETED" if r["status"] == "done" else r["status"].upper(),
        })

    risk_dist = []
    for level in ["LOW", "MEDIUM", "HIGH", "CRITICAL"]:
        cnt = sum(1 for r in recent if r["risk_level"] == level)
        risk_dist.append({"risk_level": level, "cnt": cnt})

    return {
        "total_runs": len(runs),
        "total_rules": total_rules,
        "avg_risk": avg_risk,
        "recent_runs": recent,
        "risk_dist": risk_dist,
    }

@app.get("/api/rules/history")
async def rules_history():
    all_rules = []
    rid = 1
    for run in RUNS.values():
        for r in run.get("results", []):
            all_rules.append({
                "rule_id": rid,
                "column_name": r.get("column_name", ""),
                "rule_label": r.get("rule_label", ""),
                "rule_logic": "",
                "severity": r.get("severity", ""),
                "source": r.get("source", "ai"),
            })
            rid += 1
    return all_rules

@app.delete("/api/rules/{rule_id}")
async def delete_rule(rule_id: int):
    return {"deleted": True}

@app.get("/api/runs")
async def list_runs():
    result = []
    for rid, r in RUNS.items():
        res = r.get("results", [])
        passed = sum(1 for x in res if x["status"] == "PASS")
        failed = len(res) - passed
        fail_pct = (failed / len(res) * 100) if res else 0
        risk = "LOW" if fail_pct < 10 else ("MEDIUM" if fail_pct < 30 else ("HIGH" if fail_pct < 60 else "CRITICAL"))
        result.append({
            "run_id": rid,
            "started_at": r.get("started", ""),
            "total_checks": len(res),
            "passed": passed,
            "failed": failed,
            "risk_level": risk,
            "status": "COMPLETED" if r["status"] == "done" else r["status"].upper(),
        })
    return result

# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE THREAD
# ─────────────────────────────────────────────────────────────────────────────
def _emit(run_id: str, agent: str, msg: str, level: str = "info"):
    RUNS[run_id]["log"].append({"agent": agent, "message": msg, "level": level, "type": "log"})
    print(f"[{run_id}][{agent}] {msg}")

def _run_pipeline_thread(run_id: str, session_id: str, column_rules: dict):
    try:
        RUNS[run_id]["status"] = "running"
        emit = lambda agent, msg, lvl="info": _emit(run_id, agent, msg, lvl)

        emit("Connector", "🔌 Connecting to data source...", "agent")
        df = _load_parquet(session_id, "__merged__")
        if df is None:
            tables = _load_all_tables(session_id)
            if not tables:
                raise ValueError("No data found in session")
            df = list(tables.values())[0]
        emit("Connector", f"✓ Loaded {len(df):,} rows × {len(df.columns)} columns", "success")

        emit("Profiler", "🔬 Profiling dataset...", "agent")
        from agents.profiler_gemini import profile_dataframe
        profile = profile_dataframe(df)
        issue_count = profile.get("issue_count", 0)
        emit("Profiler", f"✓ Profile complete — {issue_count} issues found", "success")

        emit("Validator", "✅ Running validation checks...", "agent")
        from agents.validator_gemini import ValidatorAgent
        validator = ValidatorAgent()
        results = validator.run(df, column_rules, emit=emit)

        passed = sum(1 for r in results if r["status"] == "PASS")
        failed = sum(1 for r in results if r["status"] == "FAIL")
        errors = sum(1 for r in results if r["status"] == "ERROR")
        emit("Validator", f"✓ Validation done — {passed} pass, {failed} fail, {errors} errors", "success")

        fixed_df = df.copy()
        fixes_applied = 0
        failures = [r for r in results if r["status"] == "FAIL"]
        if failures:
            emit("Fixer", f"🔧 Fixing {len(failures)} issues...", "agent")
            from agents.fixer_gemini import FixerAgent
            fixer = FixerAgent()
            fixed_df, fixes_applied = fixer.run(df, failures, emit=emit)
            emit("Fixer", f"✓ {fixes_applied} fixes applied", "success")
        else:
            emit("Fixer", "✓ No failures to fix", "success")

        emit("Reporter", "📊 Generating report...", "agent")
        job_dir = REPORTS_DIR / run_id
        job_dir.mkdir(parents=True, exist_ok=True)
        fixed_df.to_csv(job_dir / "cleaned_data.csv", index=False)

        # Mark fix_applied on results
        for r in results:
            if r["status"] == "FAIL" and r.get("fix_applied"):
                pass  # already set by fixer

        # Attach record-level counts to each result
        for r in results:
            col = r.get("column_name","")
            total_rows = len(df)
            affected = r.get("affected", 0)
            r["total_rows"] = total_rows
            r["failed_records"] = affected
            r["passed_records"] = total_rows - affected
        RUNS[run_id]["results"] = results
        RUNS[run_id]["fixes_applied"] = fixes_applied
        RUNS[run_id]["status"] = "done"

        total = len(results)
        pass_rate = round(passed / total * 100) if total else 0
        emit("Reporter", f"✅ Pipeline complete — {pass_rate}% pass rate, {fixes_applied} fixes applied", "success")

    except Exception as e:
        tb = traceback.format_exc()
        RUNS[run_id]["status"] = "error"
        _emit(run_id, "System", f"❌ Error: {str(e)}", "error")
        print(tb)

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def _table_meta(df: pd.DataFrame, name: str) -> dict:
    return {
        "name": name,
        "row_count": len(df),
        "col_count": len(df.columns),
        "columns": [
            {
                "column": col,
                "dtype": str(df[col].dtype),
                "null_count": int(df[col].isna().sum()),
                "unique": int(df[col].nunique()),
                "total": len(df),
                "samples": df[col].dropna().astype(str).head(3).tolist(),
            }
            for col in df.columns
        ],
    }

def _save_parquet(df: pd.DataFrame, session_id: str, table_name: str):
    """Save processed table as CSV in session root (no pyarrow needed)."""
    path = UPLOAD_DIR / session_id / f"{table_name}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)

def _load_parquet(session_id: str, table_name: str) -> Optional[pd.DataFrame]:
    """Load a processed table CSV from session root."""
    path = UPLOAD_DIR / session_id / f"{table_name}.csv"
    if path.exists():
        return pd.read_csv(path)
    return None

def _load_all_tables(session_id: str) -> dict:
    """Load only processed tables (session root *.csv), skip _raw/ and __merged__."""
    folder = UPLOAD_DIR / session_id
    if not folder.exists():
        return {}
    tables = {}
    for p in folder.glob("*.csv"):          # only session ROOT — _raw/ is a subdir, excluded
        if p.stem.startswith("__"):         # skip __merged__, __internal__ etc.
            continue
        try:
            tables[p.stem] = pd.read_csv(p)
        except Exception:
            pass
    return tables

from typing import Optional