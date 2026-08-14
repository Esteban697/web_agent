"""Tool definitions for the agent."""

import ast
from typing import Any

from langchain_core.tools import tool


@tool
def utc_now() -> str:
    """Return the current UTC timestamp in ISO format."""
    from .memory import utc_now as get_utc_now
    return get_utc_now()


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
