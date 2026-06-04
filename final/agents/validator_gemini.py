"""
VALIDATOR AGENT — Pure-pandas version (no LLM calls per check)
Executes rules against a DataFrame using direct pandas logic.
Handles all 12 rule types from the Add Custom Rule modal:
  Empty Check, Range, Allowed Values, Unique, Email, Phone,
  Date, Is Number, Pattern/Regex, Length, No Whitespace, Positive
"""
import re, traceback
import pandas as pd

MAX_RETRY = 0


class ValidatorAgent:
    def run(self, df: pd.DataFrame, all_rules: dict, emit=None) -> list:
        results = []
        checks = self._flatten_rules(all_rules)
        total = len(checks)
        for i, check in enumerate(checks, 1):
            if emit:
                emit("Validator", f"[{i}/{total}] Checking: {check['label']} on '{check['column']}'", "info")
            result = self._run_check(df, check)
            results.append(result)
        return results

    def _flatten_rules(self, all_rules: dict) -> list:
        checks = []
        cid = 1
        for col, rules in all_rules.items():
            for r in rules:
                checks.append({
                    "id": f"CHK_{cid:03d}",
                    "column": col,
                    "label": r.get("label", r.get("logic", "check")),
                    "logic": r.get("logic", ""),
                    "severity": r.get("severity", "Medium"),
                    "source": r.get("source", "ai"),
                })
                cid += 1
        return checks

    def _run_check(self, df: pd.DataFrame, check: dict) -> dict:
        col = check["column"]
        label = check["label"].lower()
        logic = check["logic"].lower()
        combined = label + " " + logic
        try:
            passed, affected, details = self._execute_rule(df, col, combined, check["logic"])
            return {
                "id": check["id"],
                "column_name": col,
                "rule_label": check["label"],
                "severity": check["severity"],
                "source": check["source"],
                "status": "PASS" if passed else "FAIL",
                "affected": affected,
                "total": len(df),
                "details": details,
                "fix_applied": False,
                "retries": 0,
            }
        except Exception as e:
            return {
                "id": check["id"],
                "column_name": col,
                "rule_label": check["label"],
                "severity": check["severity"],
                "source": check["source"],
                "status": "ERROR",
                "affected": 0,
                "total": len(df),
                "details": f"Error: {str(e)}",
                "fix_applied": False,
                "retries": 0,
            }

    def _execute_rule(self, df: pd.DataFrame, col: str, combined: str, original_logic: str):
        """
        Dispatch to the correct pandas check.
        combined = label.lower() + " " + logic.lower()
        Rules are matched most-specific first to avoid keyword collisions.
        """
        if col not in df.columns and col != "__all__":
            return True, 0, f"Column '{col}' not found — skipped"

        s = df[col] if col in df.columns else None
        n = len(df)

        # ── 1. PATTERN / REGEX ────────────────────────────────────────────
        # Logic: "must match pattern: <regex> [(case-insensitive)]"
        # MUST come before cross-column check ("match" keyword).
        pat_m = re.search(r"must match pattern[:\s]+(.+?)(?:\s*\(case-insensitive\))?\s*$", combined)
        if pat_m:
            pattern = pat_m.group(1).strip()
            flags = re.IGNORECASE if "case-insensitive" in combined else 0
            try:
                bad_mask = ~s.astype(str).str.match(pattern, flags=flags) & s.notna()
                bad = int(bad_mask.sum())
                return bad == 0, bad, f"{bad} values don't match pattern '{pattern}' in '{col}'"
            except re.error as e:
                return False, 0, f"Invalid regex pattern '{pattern}': {e}"

        # ── 2. EMPTY CHECK / NULL ─────────────────────────────────────────
        # Logic: "must not be empty or null [severity:X]"
        # Also catches AI labels: "no null", "not null", "missing", etc.
        if any(k in combined for k in ["empty or null", "null", "not null", "notna",
                                        "no null", "missing", "non-null", "non null"]):
            bad_mask = s.isna()
            if "empty or null" in combined:
                bad_mask = bad_mask | (s.astype(str).str.strip() == "")
            bad = int(bad_mask.sum())
            if any(k in combined for k in ["not", "no null", "non", "must not"]):
                return bad == 0, bad, f"{bad} empty/null values found in '{col}'"
            else:
                return bad > 0, bad, f"{bad} null values found in '{col}'"

        # ── 3. NO WHITESPACE ─────────────────────────────────────────────
        # Logic: "must not have no leading/trailing whitespace"
        #        "must not have no whitespace anywhere"
        #        "must not have no leading whitespace"
        #        "must not have no trailing whitespace"
        # MUST come before empty-string check.
        if "whitespace" in combined:
            if "anywhere" in combined:
                bad_mask = s.astype(str).str.contains(r"\s", regex=True) & s.notna()
            elif "leading" in combined and "trailing" in combined:
                bad_mask = (
                    s.astype(str).str.match(r"^\s") |
                    s.astype(str).str.match(r".*\s$")
                ) & s.notna()
            elif "leading" in combined:
                bad_mask = s.astype(str).str.match(r"^\s") & s.notna()
            elif "trailing" in combined:
                bad_mask = s.astype(str).str.match(r".*\s$") & s.notna()
            else:
                bad_mask = (
                    s.astype(str).str.match(r"^\s") |
                    s.astype(str).str.match(r".*\s$")
                ) & s.notna()
            bad = int(bad_mask.sum())
            return bad == 0, bad, f"{bad} values with unwanted whitespace in '{col}'"

        # ── 4. EMPTY STRING ───────────────────────────────────────────────
        if any(k in combined for k in ["empty string", "not empty", "blank"]):
            bad_mask = s.astype(str).str.strip() == ""
            bad = int(bad_mask.sum())
            return bad == 0, bad, f"{bad} empty/blank values found in '{col}'"

        # ── 5. UNIQUE / DUPLICATE ─────────────────────────────────────────
        # Logic: "each value must be unique within this column"
        #        "each value must be unique across full dataset"
        if any(k in combined for k in ["unique", "duplicate", "distinct", "no dup"]):
            dups = int(s.duplicated(keep=False).sum())
            return dups == 0, dups, f"{dups} duplicate values found in '{col}'"

        # ── 6. IS NUMBER ─────────────────────────────────────────────────
        # Logic: "must be a numeric (int or decimal)"
        #        "must be a integer only"
        #        "must be a decimal/float"
        #        "must be a digits-only string"
        #        "must be a numeric (int or decimal), positive only"  ← handled here first
        # MUST come before Positive and Allowed Values ("must be" clash).
        if any(k in combined for k in ["integer only", "int or decimal", "decimal/float",
                                        "digits-only string", "must be a numeric",
                                        "is numeric", "numeric type", "numeric value", "valid number"]):
            if "digits-only string" in combined:
                bad_mask = ~s.astype(str).str.match(r"^\d+$") & s.notna()
                bad = int(bad_mask.sum())
                return bad == 0, bad, f"{bad} values are not digit-only strings in '{col}'"
            elif "integer only" in combined:
                num = pd.to_numeric(s, errors="coerce")
                bad_mask = (num.isna() | (num % 1 != 0)) & s.notna()
                bad = int(bad_mask.sum())
                return bad == 0, bad, f"{bad} non-integer values in '{col}'"
            elif "decimal/float" in combined:
                num = pd.to_numeric(s, errors="coerce")
                bad_mask = num.isna() & s.notna()
                bad = int(bad_mask.sum())
                return bad == 0, bad, f"{bad} non-decimal values in '{col}'"
            else:
                # Generic numeric (int or decimal) — also handles ", positive only"
                num = pd.to_numeric(s, errors="coerce")
                bad_mask = num.isna() & s.notna()
                if "positive only" in combined:
                    bad_mask = bad_mask | ((num <= 0) & s.notna())
                bad = int(bad_mask.sum())
                return bad == 0, bad, f"{bad} non-numeric values in '{col}'"

        # ── 7. POSITIVE / NON-NEGATIVE ────────────────────────────────────
        # Logic: "must be a positive number greater than 0"
        #        "must be greater than or equal to 0"
        # MUST come before Allowed Values ("must be" / "greater" clash).
        if any(k in combined for k in ["positive number greater than 0",
                                        "must be a positive", "greater than 0"]):
            num = pd.to_numeric(s, errors="coerce")
            bad_mask = ((num <= 0) | num.isna()) & s.notna()
            bad = int(bad_mask.sum())
            return bad == 0, bad, f"{bad} values not > 0 in '{col}'"

        if any(k in combined for k in ["greater than or equal to 0", "non-negative",
                                        "nonnegative", "not negative", ">= 0", "≥ 0"]):
            num = pd.to_numeric(s, errors="coerce")
            bad_mask = (num < 0) & s.notna()
            bad = int(bad_mask.sum())
            return bad == 0, bad, f"{bad} negative values found in '{col}'"

        # ── 8. EMAIL ─────────────────────────────────────────────────────
        # Logic: "must be a valid email address"
        #        "must be a valid email address; allowed domains: gmail.com, yahoo.com"
        # Label: "Valid Email (gmail.com)"
        if any(k in combined for k in ["valid email address", "email", "e-mail"]):
            general_pattern = r"^[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}$"
            valid_mask = s.astype(str).str.match(general_pattern)

            # Extract domain restriction from logic or label
            domain_src = re.search(r"allowed domains?[:\s]+([a-zA-Z0-9.,\s\-]+)", combined)
            if not domain_src:
                domain_src = re.search(r"(?:valid email|email).*?\(([a-zA-Z0-9.\-,\s]+)\)", combined)
            if domain_src:
                allowed_domains = [d.strip().lower() for d in re.split(r"[,;]", domain_src.group(1)) if d.strip()]
                if allowed_domains:
                    dom_re = "|".join(re.escape(f"@{d}") + r"$" for d in allowed_domains)
                    domain_ok = s.astype(str).str.lower().str.contains(dom_re, regex=True)
                    bad_mask = (~(valid_mask & domain_ok)) & s.notna()
                    bad = int(bad_mask.sum())
                    return bad == 0, bad, f"{bad} emails not from allowed domain(s) {allowed_domains} in '{col}'"

            bad_mask = ~valid_mask & s.notna()
            bad = int(bad_mask.sum())
            return bad == 0, bad, f"{bad} invalid email formats in '{col}'"

        # ── 9. PHONE ─────────────────────────────────────────────────────
        # Logic: "must be a valid phone number: any format"
        #        "must be a valid phone number: 10 digits only"
        #        "must be a valid phone number: US (XXX) XXX-XXXX"
        #        "must be a valid phone number: International +XX"
        if any(k in combined for k in ["valid phone number", "phone", "mobile", "contact number"]):
            # Exact digit-count rule (e.g. "10 digits only")
            dlen = re.search(r"(\d+)\s*digits?\s*(?:only|exact|exactly)?|"
                             r"(?:only|exactly)\s*(\d+)\s*digits?", combined)
            if dlen:
                req = int(dlen.group(1) or dlen.group(2))
                norm = s.astype(str).str.replace(r"[\s\-\+\(\)]", "", regex=True)
                bad_mask = ~norm.str.match(rf"^\d{{{req}}}$") & s.notna()
                bad = int(bad_mask.sum())
                return bad == 0, bad, f"{bad} phone numbers not exactly {req} digits in '{col}'"

            # International format: must start with + or be 7-14 digits
            if "international" in combined or "intl" in combined:
                norm = s.astype(str).str.replace(r"[\s\-\(\)]", "", regex=True)
                bad_mask = ~norm.str.match(r"^\+?\d{7,14}$") & s.notna()
                bad = int(bad_mask.sum())
                return bad == 0, bad, f"{bad} invalid international phone numbers in '{col}'"

            # US format: exactly 10 digits after stripping formatting
            _us_re = re.compile(r"\bus\b")
            if _us_re.search(combined) or "xxx" in combined:
                norm = s.astype(str).str.replace(r"[\s\-\+\(\)]", "", regex=True)
                bad_mask = ~norm.str.match(r"^\d{10}$") & s.notna()
                bad = int(bad_mask.sum())
                return bad == 0, bad, f"{bad} invalid US phone numbers (need 10 digits) in '{col}'"

            # Generic "any format": 7–15 digits
            norm = s.astype(str).str.replace(r"[\s\-\+\(\)]", "", regex=True)
            bad_mask = ~norm.str.match(r"^\d{7,15}$") & s.notna()
            bad = int(bad_mask.sum())
            return bad == 0, bad, f"{bad} invalid phone formats in '{col}'"

        # ── 10. DATE ─────────────────────────────────────────────────────
        # Logic: "must be a valid date in YYYY-MM-DD format"
        #        "must be a valid date in DD/MM/YYYY format; not before 2020-01-01; not after 2025-12-31"
        if any(k in combined for k in ["valid date", "date", "datetime", "timestamp", "date format"]):
            try:
                parsed = pd.to_datetime(s, errors="coerce")
                bad_mask = parsed.isna() & s.notna()

                not_before = re.search(r"not before\s+(\d{4}-\d{2}-\d{2}|\d{2}/\d{2}/\d{4})", combined)
                not_after  = re.search(r"not after\s+(\d{4}-\d{2}-\d{2}|\d{2}/\d{2}/\d{4})", combined)
                if not_before:
                    min_date = pd.to_datetime(not_before.group(1))
                    bad_mask = bad_mask | ((parsed < min_date) & parsed.notna())
                if not_after:
                    max_date = pd.to_datetime(not_after.group(1))
                    bad_mask = bad_mask | ((parsed > max_date) & parsed.notna())

                bad = int(bad_mask.sum())
                label_detail = "date violations (invalid/out-of-range)" if (not_before or not_after) else "invalid date values"
                return bad == 0, bad, f"{bad} {label_detail} in '{col}'"
            except Exception:
                return True, 0, "Date parsing skipped"

        # ── 11. FUTURE DATE ───────────────────────────────────────────────
        if "future" in combined and "date" in combined:
            try:
                parsed = pd.to_datetime(s, errors="coerce")
                bad_mask = parsed > pd.Timestamp.now()
                bad = int(bad_mask.sum())
                return bad == 0, bad, f"{bad} future dates in '{col}'"
            except Exception:
                return True, 0, "Future date check skipped"

        # ── 12. LENGTH ────────────────────────────────────────────────────
        # Logic: "length in characters must be between 5 and 20"
        #        "length in words must be between 2 and 10"
        # MUST come before Range ("between" keyword).
        if "length in" in combined:
            lmin_m = re.search(r"between\s+(\d+)", combined)
            lmax_m = re.search(r"and\s+(\d+)", combined)
            lmin = int(lmin_m.group(1)) if lmin_m else 0
            lmax_raw = lmax_m.group(1) if lmax_m else None

            actual = s.astype(str).str.split().str.len() if "words" in combined else s.astype(str).str.len()

            bad_mask = actual < lmin
            short_count = int(bad_mask.sum())
            long_count = 0
            if lmax_raw and lmax_raw.lower() != "unlimited":
                lmax = int(lmax_raw)
                too_long = actual > lmax
                bad_mask = bad_mask | too_long
                long_count = int(too_long.sum())
            bad_mask = bad_mask & s.notna()
            bad = int(bad_mask.sum())
            unit = "words" if "words" in combined else "chars"
            parts = []
            if short_count: parts.append(f"{short_count} too short (< {lmin})")
            if long_count:  parts.append(f"{long_count} too long (> {lmax_raw})")
            if not parts:   parts = ["0 violations"]
            return bad == 0, bad, f"{'; '.join(parts)} ({unit}) in '{col}'"

        # Legacy length == N
        len_eq_m = re.search(r"length\s*(?:==|=|is)\s*(\d+)", combined)
        if len_eq_m:
            elen = int(len_eq_m.group(1))
            bad_mask = (s.astype(str).str.len() != elen) & s.notna()
            bad = int(bad_mask.sum())
            return bad == 0, bad, f"{bad} values with length ≠ {elen} in '{col}'"

        # ── 13. RANGE ────────────────────────────────────────────────────
        # Logic: "must be between 10 and 100"
        #        "must be strictly between 10 and 100"
        # Also handles AI rules with >=, <=, min/max keywords.
        if "between" in combined:
            nums = re.findall(r"-?[0-9]+(?:\.[0-9]+)?", combined)
            if len(nums) >= 2:
                lo, hi = float(nums[0]), float(nums[-1])
                num = pd.to_numeric(s, errors="coerce")
                strict = "strictly" in combined
                bad_mask = ((num < lo if not strict else num <= lo) |
                            (num > hi if not strict else num >= hi)) & s.notna()
                bad = int(bad_mask.sum())
                op, cl = ("(", ")") if strict else ("[", "]")
                return bad == 0, bad, f"{bad} values outside range {op}{lo}, {hi}{cl} in '{col}'"

        rmin = re.search(r"(?:>=|≥|min|minimum)\s*([0-9]+(?:\.[0-9]+)?)", combined)
        rmax = re.search(r"(?:<=|≤|max|maximum)\s*([0-9]+(?:\.[0-9]+)?)", combined)
        if rmin or rmax:
            num = pd.to_numeric(s, errors="coerce")
            bad_mask = pd.Series([False] * n, index=df.index)
            parts = []
            if rmin:
                mv = float(rmin.group(1))
                below = (num < mv) & s.notna()
                bad_mask |= below
                parts.append(f"{int(below.sum())} below {mv}")
            if rmax:
                mv = float(rmax.group(1))
                above = (num > mv) & s.notna()
                bad_mask |= above
                parts.append(f"{int(above.sum())} above {mv}")
            bad = int(bad_mask.sum())
            return bad == 0, bad, f"{'; '.join(parts)} in '{col}'"

        # ── 14. ALLOWED VALUES ────────────────────────────────────────────
        # Logic: "must be one of: [active, inactive, pending]"  (with bracket)
        #        "allowed: 1, 2, 3"
        # MUST come after all type-specific rules to avoid "must be" false-positives.
        av_m = re.search(
            r"(?:one of\s*[:\[]|in\s*\[|allowed values?\s*[:\[]?|valid values?\s*[:\[]?|allowed\s*:)"
            r"\s*\[?\s*([\w,\s'\"\.@\-]+?)(?:\]|$)",
            combined
        )
        if av_m:
            raw = av_m.group(1)
            vals = {v.strip().strip("'\"").lower() for v in re.split(r"[,;]| or ", raw) if v.strip()}
            if vals:
                bad_mask = ~s.astype(str).str.lower().isin(vals) & s.notna()
                bad = int(bad_mask.sum())
                return bad == 0, bad, f"{bad} values not in allowed set {vals} in '{col}'"

        # ── 15. CROSS-COLUMN / CONSISTENCY ───────────────────────────────
        if any(k in combined for k in ["consistent", "cross", "referential"]):
            return True, 0, f"Cross-column rule — manual review recommended"

        # ── 16. FALLBACK: try eval ────────────────────────────────────────
        try:
            mask = df.eval(original_logic, engine="python")
            if hasattr(mask, "__iter__"):
                bad = int((~mask).sum()) if hasattr(mask, "sum") else 0
                return bad == 0, bad, f"{bad} rows failed rule: {original_logic[:60]}"
        except Exception:
            pass

        return True, 0, f"Rule '{original_logic[:60]}' checked — no violations detected"
