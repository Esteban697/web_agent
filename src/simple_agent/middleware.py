"""Middleware for managing web journey memory recall and persistence."""

import json
from typing import Any

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import HumanMessage
from langgraph.runtime import Runtime

from .config import RECALL_LIMIT_DEFAULT, RECALL_LIMIT_HISTORY
from .memory import (
    extract_memory_with_trustcall,
    fallback_memory_doc,
    journey_namespace,
)
from .states import WebAgentState
from .utils import (
    message_text,
    is_history_query,
    memory_rank,
    dedupe_memories,
    build_memory_brief,
)


class MemoryMiddleware(AgentMiddleware):
    """Recall past web journeys before the run and save one structured memory after."""

    state_schema = WebAgentState

    async def abefore_agent(
        self,
        state: WebAgentState,
        runtime: Runtime,
    ) -> dict[str, Any] | None:
        user_id = state.get("user_id")
        if not user_id:
            return None

        messages = state.get("messages", [])
        latest_user_text = ""
        for msg in reversed(messages):
            if getattr(msg, "type", None) == "human":
                latest_user_text = message_text(getattr(msg, "content", ""))
                break

        if not latest_user_text:
            return {"recalled_web_journeys": [], "current_user_task": ""}

        store = getattr(runtime, "store", None)
        if store is None:
            return {
                "recalled_web_journeys": [],
                "current_user_task": latest_user_text,
            }

        results: list[Any] = []
        try:
            if is_history_query(latest_user_text):
                results = await store.asearch(
                    journey_namespace(user_id),
                    limit=RECALL_LIMIT_HISTORY,
                )
            else:
                results = await store.asearch(
                    journey_namespace(user_id),
                    query=latest_user_text,
                    limit=RECALL_LIMIT_DEFAULT,
                )
        except Exception as e:
            print(f"[abefore_agent] store.asearch failed: {type(e).__name__}: {e}")
            results = []

        recalled: list[dict[str, Any]] = []
        for item in results:
            value = getattr(item, "value", None) or {}
            if isinstance(value, dict):
                recalled.append(value)

        recalled = dedupe_memories(recalled)
        recalled.sort(key=lambda memory: memory_rank(memory, latest_user_text), reverse=True)

        if is_history_query(latest_user_text):
            recalled = [m for m in recalled if m.get("type") == "product_search"] or recalled
            recalled = recalled[:RECALL_LIMIT_HISTORY]
        else:
            recalled = recalled[:RECALL_LIMIT_DEFAULT]

        memory_brief = build_memory_brief(recalled)
        updated_messages = list(messages)
        if memory_brief:
            updated_messages = [
                HumanMessage(
                    content=(
                        "Memory context from earlier browsing sessions:\n\n"
                        f"{memory_brief}"
                    )
                ),
                *messages,
            ]

        return {
            "messages": updated_messages,
            "recalled_web_journeys": recalled,
            "current_user_task": latest_user_text,
        }

    async def aafter_agent(
        self,
        state: WebAgentState,
        runtime: Runtime,
    ) -> dict[str, Any] | None:
        user_id = state.get("user_id")
        task = state.get("current_user_task", "")
        if not user_id or not task:
            return None

        store = getattr(runtime, "store", None)
        if store is None:
            return None

        final_answer = ""
        messages = state.get("messages", [])
        for msg in reversed(messages):
            if getattr(msg, "type", None) == "ai":
                final_answer = message_text(getattr(msg, "content", ""))
                break

        recalled = state.get("recalled_web_journeys", []) or []
        recalled_ids = [
            m.get("memory_id")
            for m in recalled
            if isinstance(m, dict) and m.get("memory_id")
        ]

        try:
            memory_doc = await extract_memory_with_trustcall(
                user_id=user_id,
                task=task,
                final_answer=final_answer,
                recalled_ids=recalled_ids,
            )
        except Exception as e:
            print(f"[aafter_agent] Trustcall extraction failed: {type(e).__name__}: {e}")
            memory_doc = fallback_memory_doc(
                user_id=user_id,
                task=task,
                final_answer=final_answer,
                recalled_ids=recalled_ids,
            )

        try:
            await store.aput(
                journey_namespace(user_id),
                memory_doc["memory_id"],
                memory_doc,
            )
        except Exception as e:
            print(f"[aafter_agent] store.aput failed: {type(e).__name__}: {e}")
            return None

        print("Saved memory_doc:")
        print(json.dumps(memory_doc, indent=2, ensure_ascii=False))
        return None
