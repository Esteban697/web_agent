"""Tool definitions for the agent."""

import ast
import ipaddress
import socket
from typing import Any
from urllib.parse import urlparse

from langchain_core.tools import tool

_ALLOWED_SCHEMES = {"http", "https"}
_REFUSAL = (
    "Refusing to open that URL: this tool only reaches public http(s) hosts."
)


def _is_public_host(host: str) -> bool:
    """True only if every address the host resolves to is publicly routable."""
    host = host.strip("[]").lower()
    if host in {"", "localhost"} or host.endswith((".localhost", ".internal")):
        return False
    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror:
        return False
    for info in infos:
        ip = ipaddress.ip_address(info[4][0])
        if (
            ip.is_private
            or ip.is_loopback
            or ip.is_link_local
            or ip.is_reserved
            or ip.is_multicast
            or ip.is_unspecified
        ):
            return False
    return True


def _is_safe_url(url: str) -> bool:
    try:
        parsed = urlparse(url)
    except ValueError:
        return False
    if parsed.scheme not in _ALLOWED_SCHEMES or not parsed.hostname:
        return False
    return _is_public_host(parsed.hostname)


def _guarded_page(browser: Any) -> Any:
    """Return a page whose requests are aborted unless the target is public.

    Checking only the initial URL would leave redirects and subresources free to
    reach internal addresses, so every request in the context is evaluated.
    """
    context = browser.new_context()
    context.route(
        "**/*",
        lambda route: (
            route.continue_() if _is_safe_url(route.request.url) else route.abort()
        ),
    )
    return context.new_page()


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
    if not _is_safe_url(url):
        return _REFUSAL
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        return "Playwright is not installed. Run: python -m pip install playwright && python -m playwright install chromium"

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        try:
            page = _guarded_page(browser)
            page.goto(url, wait_until="domcontentloaded", timeout=30000)
            final_url = page.url
            title = page.title()
        finally:
            browser.close()
        return f"Opened: {final_url} | Title: {title}"


@tool
def read_page(url: str) -> str:
    """Open a webpage and return visible body text."""
    if not _is_safe_url(url):
        return _REFUSAL
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        return "Playwright is not installed. Run: python -m pip install playwright && python -m playwright install chromium"

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        try:
            page = _guarded_page(browser)
            page.goto(url, wait_until="domcontentloaded", timeout=30000)
            text = page.locator("body").inner_text()[:12000]
        finally:
            browser.close()
        return text
