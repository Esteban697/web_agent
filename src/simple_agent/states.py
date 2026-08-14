"""Agent state schema definition."""

from typing import Any

from langchain.agents.middleware import AgentState
from typing_extensions import NotRequired


class WebAgentState(AgentState):
    user_id: NotRequired[str]
    recalled_web_journeys: NotRequired[list[dict[str, Any]]]
    current_user_task: NotRequired[str]
