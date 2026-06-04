# 🧹 DQ Platform — Agentic Data Quality Validation System

An AI-powered, full-stack data quality validation platform built with **FastAPI** and **Google Gemini (AI Studio)**. Upload CSV/Excel files, let AI analyze schema relationships, auto-generate data quality rules, and get a cleaned dataset — all through a modern streaming UI.

---

## ✨ Features

- 📤 **Upload** CSV or Excel files (single or multi-table)
- 🔍 **Schema Analysis** — AI detects table relationships automatically
- 📋 **Rule Generation** — Gemini generates per-column data quality rules; editable by user
- ⚡ **Streaming Pipeline** — Real-time logs via SSE: Profiler → Validator → Fixer → Reporter
- 📊 **Results** — Per-rule pass/fail breakdown + downloadable cleaned CSV

---

## 🏗️ Architecture

```
final/
├── ui/
│   ├── app.py          # FastAPI backend (SSE streaming)
│   └── index.html      # Multi-step wizard frontend
├── agents/
│   ├── gemini_client.py      # Google AI Studio wrapper
│   ├── profiler_gemini.py    # Dataset profiler
│   ├── validator_gemini.py   # Rule validator (pandas)
│   ├── fixer_gemini.py       # Data fixer (Gemini-generated pandas code)
│   ├── rules_agent.py        # AI rule generator
│   └── schema_agent.py       # Multi-table schema analyzer
├── data/
│   └── sample_sales.csv      # Sample dataset for testing
├── run_ui.py           # App entry point
├── requirements.txt
├── .env.example        # Environment variable template
└── README.md
```

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
cd YOUR_REPO_NAME
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Set up environment variables

```bash
cp .env.example .env
```

Edit `.env` and add your Google AI Studio API key:

```
GOOGLE_API_KEY=your_key_here
```

> Get your free key at: https://aistudio.google.com/app/apikey

### 4. Run the app

```bash
python run_ui.py
```

Then open **http://localhost:8000** in your browser.

---

## 🧪 Testing with Sample Data

A sample dataset is included at `data/sample_sales.csv`. Upload it via the UI to see the full pipeline in action.

---

## 🤖 AI Model

Uses **Gemini 2.0 Flash** via Google AI Studio API. You can change the model in `agents/gemini_client.py`.

---

## 📦 Requirements

- Python 3.9+
- Google AI Studio API key (free tier available)

See `requirements.txt` for all Python dependencies.

---

## 📄 License

MIT License — feel free to use, modify, and distribute.
