"""Minimal LangChain agent graph with safer middleware-based long-term memory.

This version avoids wrap_tool_call/awrap_tool_call entirely.
It relies on:
- before_agent: recall similar past web journeys
- after_agent: save a structured journey/product memory

It is designed for LangGraph Agent Server environments with built-in persistence,
so it does not define a custom checkpointer or store in create_agent(...).
"""

from __future__ import annotations

import ast
import json
import os
import re
import uuid
from datetime import datetime, timezone
from typing import Any, Literal
from urllib.parse import urlparse

from pydantic import BaseModel, Field
from typing_extensions import NotRequired

from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware, AgentState
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.runtime import Runtime

DEFAULT_MODEL = os.getenv("SIMPLE_AGENT_MODEL", "anthropic:claude-sonnet-4-6")
TRUSTCALL_MEMORY_MODEL = os.getenv("TRUSTCALL_MEMORY_MODEL", DEFAULT_MODEL)


class WebAgentState(AgentState):
    user_id: NotRequired[str]
    recalled_web_journeys: NotRequired[list[dict[str, Any]]]
    current_user_task: NotRequired[str]


class ProductMemory(BaseModel):
    """Structured product extracted from a web journey result."""

    name: str = Field(description="Product name exactly as found if available")
    brand: str = Field(default="", description="Brand or vendor name")
    category: str = Field(default="", description="General category such as laptop, tablet, phone")
    price_text: str = Field(default="", description="Raw displayed price text if present")
    url: str = Field(default="", description="Product page URL if present")
    features: list[str] = Field(default_factory=list, description="Key extracted product features")


class WebJourneyMemory(BaseModel):
    """Structured memory record for one browsing journey and its discovered products."""

    memory_id: str = Field(description="Stable memory record identifier")
    type: Literal["web_journey", "product_search"] = Field(description="Type of memory")
    user_id: str = Field(description="User identifier for namespacing and filtering")
    task: str = Field(description="Original user task for this journey")
    saved_at: str = Field(description="UTC ISO timestamp when the memory was saved")
    result: str = Field(description="Concise normalized assistant answer")
    summary: str = Field(default="", description="One or two sentence summary of the journey")
    products: list[ProductMemory] = Field(default_factory=list, description="Products extracted from the journey")
    product_names: list[str] = Field(default_factory=list, description="Flattened product names for retrieval")
    brands: list[str] = Field(default_factory=list, description="Flattened brands for retrieval")
    urls: list[str] = Field(default_factory=list, description="Relevant URLs mentioned or used")
    domains: list[str] = Field(default_factory=list, description="Domains present in the journey")
    used_recalled_memories: list[str] = Field(default_factory=list, description="Memory ids used as recall context")
    success: bool = Field(description="Whether the run produced a usable answer")


_TRUSTCALL_EXTRACTOR = None
_TRUSTCALL_INIT_ERROR: Exception | None = None

try:
    from trustcall import create_extractor
except Exception as e:  # pragma: no cover
    create_extractor = None
    _TRUSTCALL_INIT_ERROR = e


def _utc_now() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


def _journey_namespace(user_id: str) -> tuple[str, ...]:
    return ("memories", user_id, "web_journeys")


def _extract_urls(text: str) -> list[str]:
    if not text:
        return []
    urls = re.findall(r'https?://[^\s)\]>\"]+', text)
    seen: set[str] = set()
    ordered: list[str] = []
    for url in urls:
        clean = url.rstrip('.,;')
        if clean not in seen:
            seen.add(clean)
            ordered.append(clean)
    return ordered


def _extract_domains(urls: list[str]) -> list[str]:
    domains: list[str] = []
    seen: set[str] = set()
    for url in urls:
        try:
            domain = urlparse(url).netloc.lower()
        except Exception:
            domain = ""
        if domain and domain not in seen:
            seen.add(domain)
            domains.append(domain)
    return domains


def _message_text(content: Any) -> str:
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
            elif isinstance(item, str):
                parts.append(item)
        return "\n".join(part for part in parts if part)

    if isinstance(content, dict):
        if content.get("type") == "text":
            return str(content.get("text", ""))
        return json.dumps(content, ensure_ascii=False)

    return str(content)


def _get_memory_extractor():
    global _TRUSTCALL_EXTRACTOR
    if _TRUSTCALL_EXTRACTOR is not None:
        return _TRUSTCALL_EXTRACTOR
    if create_extractor is None:
        raise RuntimeError(
            f"Trustcall is not available: {_TRUSTCALL_INIT_ERROR}. Install with `pip install trustcall`."
        )
    _TRUSTCALL_EXTRACTOR = create_extractor(
        TRUSTCALL_MEMORY_MODEL,
        tools=[WebJourneyMemory],
        tool_choice="WebJourneyMemory",
    )
    print("_TRUSTCALL_EXTRACTOR successfully created!")
    return _TRUSTCALL_EXTRACTOR


def _fallback_memory_doc(
    *,
    user_id: str,
    task: str,
    final_answer: str,
    recalled_ids: list[str],
) -> dict[str, Any]:
    urls = _extract_urls(final_answer)
    domains = _extract_domains(urls)
    return {
        "memory_id": str(uuid.uuid4()),
        "type": "web_journey",
        "user_id": user_id,
        "task": task,
        "saved_at": _utc_now(),
        "result": final_answer[:2000],
        "summary": final_answer[:400],
        "products": [],
        "product_names": [],
        "brands": [],
        "urls": urls[:10],
        "domains": domains,
        "used_recalled_memories": recalled_ids,
        "success": bool(final_answer),
    }


async def _extract_memory_with_trustcall(
    *,
    user_id: str,
    task: str,
    final_answer: str,
    recalled_ids: list[str],
) -> dict[str, Any]:
    extractor = _get_memory_extractor()
    base_urls = _extract_urls(final_answer)
    base_domains = _extract_domains(base_urls)
    memory_id = str(uuid.uuid4())
    saved_at = _utc_now()

    instruction = (
        "Extract a single structured WebJourneyMemory object from the assistant answer. "
        "Capture only products explicitly present in the answer. "
        "Do not invent prices, features, brands, or URLs. "
        "Set type='product_search' only when at least one product is clearly present; otherwise use 'web_journey'. "
        "Keep result concise and preserve useful retrieval fields such as product_names, brands, urls, and domains."
    )

    messages = [
        SystemMessage(content=instruction),
        HumanMessage(
            content=(
                "Build a WebJourneyMemory JSON object using this context:\n\n"
                f"user_id: {user_id}\n"
                f"task: {task}\n"
                f"memory_id: {memory_id}\n"
                f"saved_at: {saved_at}\n"
                f"used_recalled_memories: {json.dumps(recalled_ids, ensure_ascii=False)}\n"
                f"urls_detected: {json.dumps(base_urls[:10], ensure_ascii=False)}\n"
                f"domains_detected: {json.dumps(base_domains, ensure_ascii=False)}\n\n"
                "assistant_answer:\n"
                f"{final_answer[:6000]}"
            )
        ),
    ]

    result = await extractor.ainvoke({"messages": messages})

    responses = []
    if isinstance(result, dict):
        responses = result.get("responses") or result.get("response") or []
        if responses and not isinstance(responses, list):
            responses = [responses]

    structured = None
    for item in responses:
        if isinstance(item, WebJourneyMemory):
            structured = item.model_dump()
            break
        if isinstance(item, dict):
            structured = item
            break

    if not structured:
        raise ValueError("Trustcall did not return a WebJourneyMemory response")

    structured["memory_id"] = structured.get("memory_id") or memory_id
    structured["user_id"] = structured.get("user_id") or user_id
    structured["task"] = structured.get("task") or task
    structured["saved_at"] = structured.get("saved_at") or saved_at
    structured["used_recalled_memories"] = structured.get("used_recalled_memories") or recalled_ids
    structured["urls"] = structured.get("urls") or base_urls[:10]
    structured["domains"] = structured.get("domains") or base_domains
    structured["success"] = bool(structured.get("result") or final_answer)

    products = structured.get("products") or []
    product_names = []
    brands = []
    normalized_products = []
    for product in products:
        if isinstance(product, ProductMemory):
            product = product.model_dump()
        if not isinstance(product, dict):
            continue
        normalized_products.append(product)
        name = str(product.get("name", "")).strip()
        brand = str(product.get("brand", "")).strip()
        if name:
            product_names.append(name)
        if brand:
            brands.append(brand)
    structured["products"] = normalized_products
    structured["product_names"] = structured.get("product_names") or product_names
    structured["brands"] = structured.get("brands") or sorted(set(brands))
    if not structured.get("type"):
        structured["type"] = "product_search" if structured["products"] else "web_journey"

    validated = WebJourneyMemory.model_validate(structured)
    return validated.model_dump()


class WebJourneyMemoryMiddleware(AgentMiddleware):
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
                latest_user_text = _message_text(getattr(msg, "content", ""))
                break

        if not latest_user_text:
            return {"recalled_web_journeys": [], "current_user_task": ""}

        store = getattr(runtime, "store", None)
        if store is None:
            return {
                "recalled_web_journeys": [],
                "current_user_task": latest_user_text,
            }

        try:
            results = await store.asearch(
                _journey_namespace(user_id),
                query=latest_user_text,
                limit=5,
            )
        except Exception:
            results = []

        recalled: list[dict[str, Any]] = []
        for item in results:
            value = getattr(item, "value", None) or {}
            if isinstance(value, dict):
                recalled.append(value)

        return {
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
                final_answer = _message_text(getattr(msg, "content", ""))
                break

        recalled = state.get("recalled_web_journeys", []) or []
        recalled_ids = [
            m.get("memory_id")
            for m in recalled
            if isinstance(m, dict) and m.get("memory_id")
        ]

        try:
            memory_doc = await _extract_memory_with_trustcall(
                user_id=user_id,
                task=task,
                final_answer=final_answer,
                recalled_ids=recalled_ids,
            )
        except Exception as e:
            print(f"[aafter_agent] Trustcall extraction failed: {type(e).__name__}: {e}")
            memory_doc = _fallback_memory_doc(
                user_id=user_id,
                task=task,
                final_answer=final_answer,
                recalled_ids=recalled_ids,
            )

        try:
            await store.aput(
                _journey_namespace(user_id),
                memory_doc["memory_id"],
                memory_doc,
            )
        except Exception as e:
            print(f"[aafter_agent] store.aput failed: {type(e).__name__}: {e}")
            return None

        print("Saved memory_doc:")
        print(json.dumps(memory_doc, indent=2, ensure_ascii=False))
        return None


@tool
def utc_now() -> str:
    """Return the current UTC timestamp in ISO format."""
    return _utc_now()


@tool
def calculator(expression: str) -> str:
    """Evaluate a simple arithmetic expression safely.

    Supported operators: +, -, *, /, %, ** and parentheses.
    """
    parsed = ast.parse(expression, mode="eval")
    allowed_nodes = (
        ast.Expression,
        ast.BinOp,
        ast.UnaryOp,
        ast.Constant,
        ast.Add,
        ast.Sub,
        ast.Mult,
        ast.Div,
        ast.Mod,
        ast.Pow,
        ast.USub,
        ast.UAdd,
        ast.Load,
    )

    for node in ast.walk(parsed):
        if not isinstance(node, allowed_nodes):
            raise ValueError("Expression contains unsupported syntax")

    result: Any = eval(compile(parsed, "", "eval"), {"__builtins__": {}}, {})
    return str(result)


@tool
def open_page(url: str) -> str:
    """Open a webpage in Chromium and return the final URL and title."""
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        return "Playwright is not installed. Run: python -m pip install playwright && python -m playwright install chromium"

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto(url, wait_until="domcontentloaded", timeout=30000)
        final_url = page.url
        title = page.title()
        browser.close()
        return f"Opened: {final_url} | Title: {title}"


@tool
def read_page(url: str) -> str:
    """Open a webpage and return visible body text."""
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        return "Playwright is not installed. Run: python -m pip install playwright && python -m playwright install chromium"

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto(url, wait_until="domcontentloaded", timeout=30000)
        text = page.locator("body").inner_text()[:12000]
        browser.close()
        return text


graph = create_agent(
    model=DEFAULT_MODEL,
    tools=[utc_now, calculator, open_page, read_page],
    middleware=[WebJourneyMemoryMiddleware()],
    state_schema=WebAgentState,
    system_prompt=(
        "You are a concise assistant. "
        "Use tools when they add factual precision, then return a direct answer. "
        "If recalled_web_journeys exists in state, use it only as a hint and verify with current browsing."
    ),
    name="simple_agent",
)