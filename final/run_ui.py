"""
Start the DQ Platform UI.
Run: python run_ui.py
Then open: http://localhost:8000
"""
import subprocess, sys

if __name__ == "__main__":
    subprocess.run([
        sys.executable, "-m", "uvicorn",
        "ui.app:app",
        "--reload",
        "--host", "0.0.0.0",
        "--port", "8000",
    ])
