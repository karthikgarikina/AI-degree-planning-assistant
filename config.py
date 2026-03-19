from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    llm_provider: str = os.getenv("LLM_PROVIDER", "ollama")
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")
    ollama_model: str = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
    ollama_timeout_seconds: int = int(os.getenv("OLLAMA_TIMEOUT_SECONDS", "60"))
    anthropic_api_key: str = os.getenv("ANTHROPIC_API_KEY", "")
    mcp_server_url: str = os.getenv("MCP_SERVER_URL", "http://mock-mcp-server:8000")
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    agent_max_steps: int = int(os.getenv("AGENT_MAX_STEPS", "10"))
    catalog_path: str = os.getenv("COURSE_CATALOG_PATH", "data/course_catalog.json")
    requirements_path: str = os.getenv("PROGRAM_REQUIREMENTS_PATH", "data/program_requirements.json")
    output_dir: str = os.getenv("OUTPUT_DIR", "output")


settings = Settings()
