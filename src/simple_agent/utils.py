"""Utility functions for text extraction, parsing, and memory formatting."""

import json
import re
from typing import Any
from urllib.parse import urlparse


def extract_urls(text: str) -> list[str]:
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


def extract_domains(urls: list[str]) -> list[str]:
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


def message_text(content: Any) -> str:
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


def is_history_query(text: str) -> bool:
    t = text.lower()
    cues = (
        "earlier",
        "before",
        "previous",
        "history",
        "found earlier",
        "what products",
        "all products",
        "products found",
        "have been found",
        "we found",
        "we saw",
        "previous thread",
    )
    return any(cue in t for cue in cues)


def memory_rank(memory: dict[str, Any], query: str) -> tuple[int, int, str]:
    q = query.lower()
    type_bonus = 2 if memory.get("type") == "product_search" else 0
    hit_bonus = 0

    for name in memory.get("product_names", []) or []:
        if str(name).lower() in q or any(token in q for token in str(name).lower().split()):
            hit_bonus += 3

    for brand in memory.get("brands", []) or []:
        if str(brand).lower() in q:
            hit_bonus += 2

    haystacks = [
        str(memory.get("task", "")),
        str(memory.get("summary", "")),
        str(memory.get("search_text", "")),
        str(memory.get("result", ""))[:500],
    ]
    for text in haystacks:
        low = text.lower()
        for token in set(re.findall(r"[a-z0-9]{3,}", q)):
            if token in low:
                hit_bonus += 1

    return (
        type_bonus + hit_bonus,
        len(memory.get("product_names", []) or []),
        str(memory.get("saved_at", "")),
    )


def dedupe_memories(memories: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for memory in memories:
        memory_id = str(memory.get("memory_id", "")).strip()
        if memory_id and memory_id in seen_ids:
            continue
        if memory_id:
            seen_ids.add(memory_id)
        deduped.append(memory)
    return deduped


def fallback_search_text(memory: dict[str, Any]) -> str:
    products = memory.get("products") or []
    product_bits: list[str] = []
    for p in products:
        if not isinstance(p, dict):
            continue
        bit = " | ".join(
            part
            for part in [
                str(p.get("name", "")).strip(),
                str(p.get("brand", "")).strip(),
                str(p.get("category", "")).strip(),
                str(p.get("price_text", "")).strip(),
            ]
            if part
        )
        if bit:
            product_bits.append(bit)

    parts = [
        str(memory.get("task", "")).strip(),
        str(memory.get("summary", "")).strip(),
        ", ".join(memory.get("product_names", []) or []),
        ", ".join(memory.get("brands", []) or []),
        ", ".join(memory.get("domains", []) or []),
        " ; ".join(product_bits[:10]),
    ]
    return " || ".join(part for part in parts if part)


def build_memory_brief(recalled: list[dict[str, Any]]) -> str:
    if not recalled:
        return ""

    lines: list[str] = [
        "Recalled memory from earlier browsing sessions:",
    ]

    for i, memory in enumerate(recalled[:5], start=1):
        task = str(memory.get("task", "")).strip()
        summary = str(memory.get("summary", "")).strip()
        product_names = memory.get("product_names", []) or []
        brands = memory.get("brands", []) or []

        lines.append(f"{i}. Task: {task or 'Unknown task'}")

        if summary:
            lines.append(f"   Summary: {summary}")

        if product_names:
            product_text = ", ".join(str(x) for x in product_names[:10] if str(x).strip())
            if product_text:
                lines.append(f"   Products: {product_text}")

        if brands:
            brand_text = ", ".join(str(x) for x in brands[:10] if str(x).strip())
            if brand_text:
                lines.append(f"   Brands: {brand_text}")

    return "\n".join(lines)
