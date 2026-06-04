# DQ Platform — Agentic Data Quality Validation System
## Google AI Studio Edition (Advanced UI)

### Overview
This is a full-stack data quality validation platform that combines:
- **Advanced UI** from DQVA_v2_fixed (multi-step wizard, SSE streaming logs, schema analysis)
- **Google AI Studio API** (Gemini 2.0 Flash) for all AI agent calls

### Setup
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Set your Google AI Studio API key in `.env`:
   ```
   GOOGLE_API_KEY=your_key_here
   ```
   Get your key at: https://aistudio.google.com/app/apikey

3. Run the UI:
   ```bash
   python run_ui.py
   ```
   Then open http://localhost:8000

### Architecture
- `ui/app.py` — FastAPI backend (v2 advanced backend with SSE streaming)
- `ui/index.html` — Advanced multi-step frontend UI
- `agents/gemini_client.py` — Google AI Studio wrapper
- `agents/profiler_gemini.py` — Dataset profiler (Gemini)
- `agents/validator_gemini.py` — Rule validator (pure pandas, no LLM)
- `agents/fixer_gemini.py` — Data fixer (Gemini generates pandas code)
- `agents/rules_agent.py` — AI rule generator (Gemini)
- `agents/schema_agent.py` — Schema analyzer for multi-table uploads (Gemini)

### Pipeline Flow
1. **Upload** — Upload 1+ CSV/Excel files (or connect to SQL Server)
2. **Schema Analysis** — AI analyzes table relationships
3. **Column Rules** — AI generates DQ rules per column; user can edit/add
4. **Run** — Streaming pipeline: Profiler → Validator → Fixer → Reporter
5. **Results** — View per-rule pass/fail; download cleaned CSV with analysis

### API Key
Uses `GOOGLE_API_KEY` (Google AI Studio). Compatible with any Gemini model.
Default model: `gemini-2.0-flash`
