"""
FIXER AGENT — Gemini version
Generates fix code for failed checks, applies them to DataFrame.
"""
import re, io
import pandas as pd
from agents.gemini_client import generate

SYSTEM = """You are a pandas data-cleaning code writer.
Write a Python function called apply_fix(df) that:
- Accepts a pandas DataFrame df
- Fixes the specific issue described
- Returns the fixed DataFrame
- Is safe — does not destroy data, only cleans/repairs
Return ONLY raw Python code. No markdown. No explanation.
Start with: def apply_fix(df):"""


class FixerAgent:
    def run(self, df: pd.DataFrame, failures: list, emit=None) -> pd.DataFrame:
        fixed_df = df.copy()
        fixes_applied = 0
        for f in failures:
            if emit:
                emit("Fixer", f"Generating fix for: {f['rule_label']} on '{f['column_name']}'", "info")
            try:
                code = self._gen_fix(df, f)
                ns = {}
                exec(compile(code, "<dq_fix>", "exec"), {"pd": pd, "__builtins__": __builtins__}, ns)
                fn = ns.get("apply_fix")
                if fn:
                    fixed_df = fn(fixed_df)
                    f["fix_applied"] = True
                    fixes_applied += 1
                    if emit:
                        emit("Fixer", f"✓ Fix applied for {f['column_name']}", "success")
            except Exception as e:
                if emit:
                    emit("Fixer", f"⚠ Fix failed for {f['column_name']}: {e}", "warn")
        return fixed_df, fixes_applied

    def _gen_fix(self, df: pd.DataFrame, failure: dict) -> str:
        prompt = (
            f"DataFrame columns: {df.columns.tolist()}\n"
            f"Dtypes: { {c: str(t) for c, t in df.dtypes.items()} }\n\n"
            f"Generate apply_fix(df) to fix:\n"
            f"  Column: {failure['column_name']}\n"
            f"  Issue: {failure['rule_label']}\n"
            f"  Details: {failure.get('details', '')[:200]}\n"
            f"  Affected rows: {failure.get('affected', 0)}"
        )
        code = generate(prompt, system=SYSTEM)
        code = re.sub(r"```(?:python)?", "", code).replace("```", "").strip()
        return code
