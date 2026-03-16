import os
from dotenv import load_dotenv

load_dotenv()

class Settings:

    LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")
    OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b")

    AGENT_MAX_STEPS = int(os.getenv("AGENT_MAX_STEPS", "10"))

    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")


settings = Settings()