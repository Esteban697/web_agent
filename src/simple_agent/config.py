"""Configuration constants and environment variables."""

import os

DEFAULT_MODEL = os.getenv("SIMPLE_AGENT_MODEL", "anthropic:claude-sonnet-4-6")
TRUSTCALL_MEMORY_MODEL = os.getenv("TRUSTCALL_MEMORY_MODEL", DEFAULT_MODEL)
RECALL_LIMIT_DEFAULT = int(os.getenv("MEMORY_RECALL_LIMIT", "8"))
RECALL_LIMIT_HISTORY = int(os.getenv("MEMORY_HISTORY_LIMIT", "20"))
