"""Frontend settings. Holds no secrets: the backend owns every API key."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

API_PORT = int(os.getenv("API_PORT", "8000"))
BACKEND_URL = os.getenv("NEXUS_BACKEND_URL", f"http://127.0.0.1:{API_PORT}")
API_URL = f"{BACKEND_URL}/api/v1"

# Start uvicorn automatically when the frontend starts and the backend isn't already running
AUTO_START_BACKEND = os.getenv("NEXUS_AUTO_START_BACKEND", "true").lower() == "true"
BACKEND_START_TIMEOUT_S = 90  # first start imports LangChain, Pinecone, etc.
BACKEND_LOG = ROOT / "logs" / "backend.log"

REQUEST_TIMEOUT_S = 180  # agent answers with several tool rounds can take a while
MODELS = {"glm": "GLM 5.3 (Hugging Face)", "kimi": "Kimi K3 (Hugging Face)", "openai": "OpenAI"}
