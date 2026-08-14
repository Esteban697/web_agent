"""Data models for web journey memory and product extraction."""

from typing import Literal

from pydantic import BaseModel, Field, ConfigDict


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

    model_config = ConfigDict(title="WebJourneyMemory")

    memory_id: str = Field(description="Stable memory record identifier")
    type: Literal["web_journey", "product_search"] = Field(description="Type of memory")
    user_id: str = Field(description="User identifier for namespacing and filtering")
    task: str = Field(description="Original user task for this journey")
    saved_at: str = Field(description="UTC ISO timestamp when the memory was saved")
    result: str = Field(description="Concise normalized assistant answer")
    summary: str = Field(default="", description="One or two sentence summary of the journey")
    search_text: str = Field(
        default="",
        description="Compact retrieval text for semantic search combining task, brands, product names, categories, store, country, and summary",
    )
    products: list[ProductMemory] = Field(default_factory=list, description="Products extracted from the journey")
    product_names: list[str] = Field(default_factory=list, description="Flattened product names for retrieval")
    brands: list[str] = Field(default_factory=list, description="Flattened brands for retrieval")
    urls: list[str] = Field(default_factory=list, description="Relevant URLs mentioned or used")
    domains: list[str] = Field(default_factory=list, description="Domains present in the journey")
    used_recalled_memories: list[str] = Field(default_factory=list, description="Memory ids used as recall context")
    success: bool = Field(description="Whether the run produced a usable answer")
