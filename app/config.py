"""Application configuration.

Loads environment variables from a local ``.env`` file and exposes typed
settings and filesystem paths used across the app. This module is fully
functional (not a stub).
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# Load variables from a .env file at the project root, if present.
# Existing environment variables take precedence over the file.
load_dotenv()

# --- Model configuration --------------------------------------------------

#: Gemini API key, read from the environment (see ``.env.example``).
GEMINI_API_KEY: str | None = os.getenv("GEMINI_API_KEY")

#: LiteLLM-compatible model identifier for Gemini (conversational agent).
#: Defaults to gemini-3.1-flash-lite: fast, reliable, and supports function
#: calling. (gemini-3.5-flash also works but was frequently 503 "high demand".)
MODEL_NAME: str = os.getenv("MODEL_NAME", "gemini/gemini-3.1-flash-lite")

#: Cheaper/faster Gemini model used for the one-time inventory enrichment pass.
#: Note: the "gemini-3.5-flash-lite" string 404s on the API (no such model);
#: gemini-2.5-flash-lite is a working lite variant, so we default to it.
ENRICH_MODEL_NAME: str = os.getenv("ENRICH_MODEL_NAME", "gemini/gemini-2.5-flash-lite")

# --- Filesystem paths -----------------------------------------------------

#: Project root directory (…/Dubz_AI_Bot).
BASE_DIR: Path = Path(__file__).resolve().parent.parent

#: Directory holding the inventory dataset and generated data files.
DATA_DIR: Path = BASE_DIR / "data"

#: Path to the car inventory workbook (xlsx dataset placed here by the user).
INVENTORY_PATH: Path = DATA_DIR / "inventory.xlsx"

#: CSV file where captured sales leads are appended.
LEADS_PATH: Path = DATA_DIR / "leads.csv"

#: SQLite database storing per-user profiles and memory.
DB_PATH: Path = DATA_DIR / "dubz.db"
