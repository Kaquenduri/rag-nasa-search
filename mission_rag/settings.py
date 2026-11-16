import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
SECRET_KEY = os.environ.get("DJANGO_SECRET_KEY") or os.urandom(32).hex()
DEBUG = False
ALLOWED_HOSTS = [host.strip() for host in os.getenv("DJANGO_ALLOWED_HOSTS", "localhost,127.0.0.1,[::1]").split(",") if host.strip()]
if render_host := os.getenv("RENDER_EXTERNAL_HOSTNAME", "").strip():
    ALLOWED_HOSTS.append(render_host)
ROOT_URLCONF = "mission_rag.urls"
WSGI_APPLICATION = "mission_rag.wsgi.application"
INSTALLED_APPS = ["mission_rag"]
MIDDLEWARE = ["django.middleware.security.SecurityMiddleware", "django.middleware.common.CommonMiddleware"]
DATA_UPLOAD_MAX_MEMORY_SIZE = 16384
DEFAULT_CHARSET = "utf-8"

OLLAMA_URL = os.getenv("OLLAMA_URL", "https://scouting-flatten-arrival.ngrok-free.dev/api/generate")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3.5:2b")
RAG_API_TOKEN = os.getenv("RAG_API_TOKEN", "")

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "").strip()
GOOGLE_EMBED_MODEL = os.getenv("GOOGLE_EMBED_MODEL", "gemini-embedding-001").strip()
RAG_EMBED_MIN_SCORE = float(os.getenv("RAG_EMBED_MIN_SCORE", "0.25"))
