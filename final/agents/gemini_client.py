"""
Google AI Studio API wrapper – uses google-generativeai SDK.
Compatible with Google AI Studio API keys (GOOGLE_API_KEY).
"""
import os
import google.generativeai as genai
from dotenv import load_dotenv
load_dotenv()

GEMINI_MODEL = "gemini-2.0-flash"   # Google AI Studio model

def generate(prompt: str, system: str = None, json_mode: bool = False) -> str:
    api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY not set in environment")

    genai.configure(api_key=api_key)

    generation_config = {}
    if json_mode:
        generation_config["response_mime_type"] = "application/json"

    model = genai.GenerativeModel(
        model_name=GEMINI_MODEL,
        system_instruction=system if system else None,
        generation_config=generation_config if generation_config else None,
    )

    response = model.generate_content(prompt)
    return response.text.strip()
