"""Minimal LangChain agent graph for deployment."""

from __future__ import annotations

import ast
import os
from datetime import datetime, timezone
from typing import Any

from langchain.agents import create_agent
from langchain_core.tools import tool

from playwright.sync_api import sync_playwright
from langchain_core.tools import tool

DEFAULT_MODEL = os.getenv("SIMPLE_AGENT_MODEL", "anthropic:claude-sonnet-4-6")

_playwright = None
_browser = None
_page = None

def get_page():
    global _playwright, _browser, _page
    if _page is None:
        _playwright = sync_playwright().start()
        _browser = _playwright.chromium.launch(headless=True)
        _page = _browser.new_page()
    return _page

@tool
def open_page(url: str) -> str:
    """Open a webpage and wait until the DOM is ready."""
    page = get_page()
    page.goto(url, wait_until="domcontentloaded", timeout=30000)
    return f"Opened {page.url}"

@tool
def click(selector: str) -> str:
    """Click an element by CSS selector."""
    page = get_page()
    page.click(selector, timeout=10000)
    return f"Clicked {selector}"

@tool
def read_page() -> str:
    """Return the visible page text."""
    page = get_page()
    return page.locator("body").inner_text()[:12000]


@tool
def utc_now() -> str:
    """Return the current UTC timestamp in ISO format."""
    return datetime.now(tz=timezone.utc).isoformat()


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

    result: Any = eval(compile(parsed, "<calculator>", "eval"), {"__builtins__": {}}, {})
    return str(result)


graph = create_agent(
    model=DEFAULT_MODEL,
    tools=[utc_now, calculator, open_page, click, read_page],
    system_prompt=(
        "You are a concise assistant. "
        "Use tools when they add factual precision, then return a direct answer."
    ),
    name="simple_agent",
)
