"""Minimal LangChain agent graph with safer middleware-based long-term memory.

This version avoids wrap_tool_call/awrap_tool_call entirely.
It relies on:
- before_agent: recall similar past web journeys
- after_agent: save a structured journey/product memory

It is designed for LangGraph Agent Server environments with built-in persistence,
so it does not define a custom checkpointer or store in create_agent(...).
"""

from __future__ import annotations
import traceback
import ast
import json
import os
import re
import uuid
from datetime import datetime, timezone
from typing import Any
from urllib.parse import urlparse

from typing_extensions import NotRequired

import langchain
from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware, AgentState
from langchain_core.tools import tool
from langgraph.runtime import Runtime
from langgraph.store.base import BaseStore

#Prints out relevant data to the terminal for better debugging
langchain.debug = True

DEFAULT_MODEL = os.getenv("SIMPLE_AGENT_MODEL", "anthropic:claude-sonnet-4-6")


class WebAgentState(AgentState):
    user_id: NotRequired[str]
    recalled_web_journeys: NotRequired[list[dict[str, Any]]]
    current_user_task: NotRequired[str]


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
        clean = url.rstrip(".,;")
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


def _extract_product_candidates(answer: str) -> list[dict[str, Any]]:
    products: list[dict[str, Any]] = []
    if not answer:
        return products

    lines = [line.strip(" -*\t") for line in answer.splitlines() if line.strip()]
    feature_markers = (
        "feature",
        "features",
        "spec",
        "specs",
        "processor",
        "cpu",
        "gpu",
        "ram",
        "memory",
        "storage",
        "display",
        "screen",
        "battery",
        "camera",
        "price",
    )

    current: dict[str, Any] | None = None
    for line in lines:
        if ":" in line and len(line.split(":", 1)[0].split()) <= 4:
            label, value = [part.strip() for part in line.split(":", 1)]
            key = label.lower()

            if key in {"product", "name", "model", "item"} and value:
                if current:
                    products.append(current)
                current = {
                    "name": value,
                    "brand": value.split()[0] if value else "",
                    "features": [],
                }
                continue

            if current and any(marker in key for marker in feature_markers) and value:
                current.setdefault("features", []).append(f"{label}: {value}")
                continue

        if line.startswith(("-", "•")):
            bullet = line.lstrip("-• ").strip()
            if current and bullet:
                current.setdefault("features", []).append(bullet)

    if current:
        products.append(current)

    if products:
        return products[:10]

    sentence_products: list[dict[str, Any]] = []
    sentences = re.split(r"(?<=[.!?])\s+", answer)
    product_pattern = re.compile(
        r"\b([A-Z][A-Za-z0-9]+(?:\s+[A-Z0-9][A-Za-z0-9+\-/]*){0,5})\b"
    )

    for sentence in sentences:
        if not any(marker in sentence.lower() for marker in feature_markers):
            continue
        matches = product_pattern.findall(sentence)
        if not matches:
            continue
        name = matches[0].strip()
        sentence_products.append(
            {
                "name": name,
                "brand": name.split()[0] if name else "",
                "features": [sentence.strip()],
            }
        )

    deduped: list[dict[str, Any]] = []
    seen_names: set[str] = set()
    for product in sentence_products:
        name = product.get("name", "")
        if name and name not in seen_names:
            seen_names.add(name)
            deduped.append(product)

    return deduped[:10]

def _message_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(item.get("text", ""))
            elif isinstance(item, str):
                parts.append(item)
        return "\n".join(part for part in parts if part)
    return str(content)


class WebJourneyMemoryMiddleware(AgentMiddleware):
    """Recall past web journeys before the run and save one structured memory after."""

    state_schema = WebAgentState

    async def abefore_agent(
        self,
        state: WebAgentState,
        runtime: Runtime,
    ) -> dict[str, Any] | None:
        print("[before_agent] start")
        user_id = state.get("user_id")
        print(f"[before_agent] user_id={user_id!r}")
        if not user_id:
            print("[before_agent] missing user_id; skipping recall")
            return None

        messages = state.get("messages", [])
        print(f"[before_agent] messages_count={len(messages)}")
        latest_user_text = ""
        for msg in reversed(messages):
            if getattr(msg, "type", None) == "human":
                latest_user_text = str(getattr(msg, "content", ""))
                break

        print(f"[before_agent] latest_user_text={latest_user_text!r}")
        if not latest_user_text:
            print("[before_agent] no latest user text found")
            return {"recalled_web_journeys": [], "current_user_task": ""}

        store = getattr(runtime, "store", None)
        print(f"[before_agent] runtime_has_store={store is not None}")
        if store is None:
            print("[before_agent] runtime.store is None; returning empty recall")
            return {
                "recalled_web_journeys": [],
                "current_user_task": latest_user_text,
            }

        namespace = _journey_namespace(user_id)
        print(f"[before_agent] namespace={namespace}")

        try:
            results = await store.asearch(
                _journey_namespace(user_id),
                limit=20,
            )
            print(f"[before_agent] raw_results_count={len(results)}")
        except Exception as e:
            print(f"[before_agent] store.search exception: {type(e).__name__}: {e}")
            print(traceback.format_exc())
            results = []

        recalled: list[dict[str, Any]] = []
        for idx, item in enumerate(results):
            value = getattr(item, "value", None) or {}
            key = getattr(item, "key", None)
            print(
                f"[before_agent] result[{idx}] key={key!r} value_type={type(value).__name__}"
            )
            if isinstance(value, dict):
                recalled.append(value)

        print(f"[before_agent] recalled_count={len(recalled)}")
        if recalled:
            memory_ids = [m.get("memory_id") for m in recalled if isinstance(m, dict)]
            print(f"[before_agent] recalled_memory_ids={memory_ids}")

        return {
            "recalled_web_journeys": recalled,
            "current_user_task": latest_user_text,
        }

    async def aafter_agent(
        self,
        state: WebAgentState,
        runtime: Runtime,
    ) -> dict[str, Any] | None:
        print("[after_agent] start")
        user_id = state.get("user_id")
        task = state.get("current_user_task", "")
        print(f"[after_agent] user_id={user_id!r}")
        print(f"[after_agent] task={task!r}")
        if not user_id or not task:
            print("[after_agent] missing user_id or task; skipping save")
            return None

        store = getattr(runtime, "store", None)
        print(f"[after_agent] runtime_has_store={store is not None}")
        if store is None:
            print("[after_agent] runtime.store is None; skipping save")
            return None

        final_answer = ""
        messages = state.get("messages", [])
        print(f"[after_agent] messages_count={len(messages)}")
        for msg in reversed(messages):
            if getattr(msg, "type", None) == "ai":
                final_answer = _message_text(getattr(msg, "content", ""))
                break

        print(f"[after_agent] final_answer_preview={final_answer[:300]!r}")

        recalled = state.get("recalled_web_journeys", []) or []
        recalled_ids = [
            m.get("memory_id")
            for m in recalled
            if isinstance(m, dict) and m.get("memory_id")
        ]
        print(f"[after_agent] recalled_ids={recalled_ids}")

        urls = _extract_urls(final_answer)
        domains = _extract_domains(urls)
        products = _extract_product_candidates(final_answer)
        print(f"[after_agent] urls={urls[:10]}")
        print(f"[after_agent] domains={domains}")
        print(f"[after_agent] products_count={len(products)}")
        if products:
            print(
                f"[after_agent] product_names={[p.get('name') for p in products if isinstance(p, dict)]}"
            )

        memory_type = "product_search" if products else "web_journey"
        memory_doc = {
            "memory_id": str(uuid.uuid4()),
            "type": memory_type,
            "task": task,
            "saved_at": _utc_now(),
            "result": final_answer[:2000],
            "products": products,
            "product_names": [p.get("name", "") for p in products if p.get("name")],
            "brands": sorted({p.get("brand", "") for p in products if p.get("brand")}),
            "urls": urls[:10],
            "domains": domains,
            "used_recalled_memories": recalled_ids,
            "success": bool(final_answer),
        }

        namespace = _journey_namespace(user_id)
        print(f"[after_agent] namespace={namespace}")
        print(f"[after_agent] memory_id={memory_doc['memory_id']}")

        try:
            await store.aput(
                namespace,
                memory_doc["memory_id"],
                memory_doc,
            )
            print("[after_agent] store.put success")
        except Exception as e:
            print(f"[after_agent] store.put exception: {type(e).__name__}: {e}")
            print(traceback.format_exc())
            return None

        print("[after_agent] saved memory_doc:")
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