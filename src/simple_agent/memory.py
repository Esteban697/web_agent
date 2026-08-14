"""Memory extraction and persistence logic using Trustcall."""

import json
import uuid
from datetime import datetime, timezone
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from .config import TRUSTCALL_MEMORY_MODEL
from .models import ProductMemory, WebJourneyMemory
from .utils import extract_urls, extract_domains, fallback_search_text

_TRUSTCALL_EXTRACTOR = None
_TRUSTCALL_INIT_ERROR: Exception | None = None

try:
    from trustcall import create_extractor
except Exception as e:  # pragma: no cover
    create_extractor = None
    _TRUSTCALL_INIT_ERROR = e


def utc_now() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


def journey_namespace(user_id: str) -> tuple[str, ...]:
    return ("memories", user_id, "web_journeys")


def get_memory_extractor():
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
    return _TRUSTCALL_EXTRACTOR


def fallback_memory_doc(
    *,
    user_id: str,
    task: str,
    final_answer: str,
    recalled_ids: list[str],
) -> dict[str, Any]:
    urls = extract_urls(final_answer)
    domains = extract_domains(urls)
    memory = {
        "memory_id": str(uuid.uuid4()),
        "type": "web_journey",
        "user_id": user_id,
        "task": task,
        "saved_at": utc_now(),
        "result": final_answer[:2000],
        "summary": final_answer[:400],
        "search_text": "",
        "products": [],
        "product_names": [],
        "brands": [],
        "urls": urls[:10],
        "domains": domains,
        "used_recalled_memories": recalled_ids,
        "success": bool(final_answer),
    }
    memory["search_text"] = fallback_search_text(memory)
    return memory


async def extract_memory_with_trustcall(
    *,
    user_id: str,
    task: str,
    final_answer: str,
    recalled_ids: list[str],
) -> dict[str, Any]:
    extractor = get_memory_extractor()
    base_urls = extract_urls(final_answer)
    base_domains = extract_domains(base_urls)
    memory_id = str(uuid.uuid4())
    saved_at = utc_now()

    instruction = (
        "Extract a single structured WebJourneyMemory object from the assistant answer. "
        "Capture only products explicitly present in the answer. "
        "Do not invent prices, features, brands, or URLs. "
        "Set type='product_search' only when at least one product is clearly present; otherwise use 'web_journey'. "
        "Populate search_text as a compact retrieval string that combines task, summary, product names, brands, categories, country/store clues, domains, and notable price text. "
        "Keep search_text dense, factual, and under 300 characters if possible."
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
    if not structured.get("search_text"):
        structured["search_text"] = fallback_search_text(structured)

    validated = WebJourneyMemory.model_validate(structured)
    return validated.model_dump()
