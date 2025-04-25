"""Configuration settings for the Nomad Travel Agent."""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# --- API Keys ---
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
OPENWEATHERMAP_API_KEY = os.getenv("OPENWEATHERMAP_API_KEY")
UNSPLASH_ACCESS_KEY = os.getenv("UNSPLASH_ACCESS_KEY")

# --- Modes ---
INTERACTIVE_MODE = os.getenv("INTERACTIVE_MODE", "False").lower() == "true"

# --- Model Names ---
SUGGESTION_AGENT_MODEL = "gemini-2.0-flash"
PHASE2_MODEL_NAME = "gemini-2.0-flash"
QNA_MODEL_NAME = "gemini-2.0-flash"
EMBEDDING_MODEL_NAME = "models/text-embedding-004"

# --- Vector Store ---
CHROMA_CACHE_DIR = "./chroma_db_persistent_cache"
COLLECTION_NAME = "wikivoyage_pois_gemini_v1"

# --- Data Source ---
WIKIVOYAGE_CSV_URL = "https://github.com/wikivoyage/wikivoyage.github.io/blob/master/wikivoyage-listings-en.csv?raw=true"

# --- Placeholders ---
PLACEHOLDER_IMAGE_URL = "https://images.unsplash.com/photo-1500835556837-99ac94a94552"

# --- Demo Data ---
DEMO_PREFERENCE_ANSWERS = [
    "I'm looking for a relaxing vibe, maybe somewhere quiet.",
    "Mainly spa treatments and maybe some light reading by a pool.",
    "I definitely prefer warm and sunny weather.",
    "Let's go with mid-range to luxury.",
    "Marrakech",
]
MAX_DEMO_QUESTIONS = 2

# --- Other Constants ---
RAG_N_RESULTS = 5
RECURSION_LIMIT = 150
TOOL_RETRY_ATTEMPTS = 2

print(f"Running in {'INTERACTIVE' if INTERACTIVE_MODE else 'DEMO'} mode.")
