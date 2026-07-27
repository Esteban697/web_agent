# Web Agent in LangGraph

Web Agent is a flexible up-to-date implementation in LangGraph capable of browsing the web to find products for the user.
It is basically a flexible Product Reseatch Assistant that can save the general information about a product in the web store the user describes in the request.
It can go to a specific URL or find it based on the user description, click and read through the content to find relevant information about the product or products it finds there.
It is deployable in LangSmith Studio to easily test and debug.

## Features:

**Short term memory:**
The agent will remember the previous message inside the same conversation when using the same value in "User Id" field (required).

**Long term memory:**
The Postgres storage the *langgraph dev* command adds when packaging the LangGraph project saves the product information and the previous requests in a JSON structure that can be easily read and used for future features.

**Trustcall:**
This library is used for the LLM to have the capability to read through the browsing results and extract the values of the JSON fields in a more accurate way than just instructing it as part of the system prompt.

## Quickstart

1. Sync the project with `uv`:

```bash
uv sync --dev
```

2. Configure environment:

```bash
cp .env.example .env
```

3. Run locally:

```bash
uv run langgraph dev
```

Optional `make` wrappers:

```bash
make dev
make run
```

## Tests and lint

```bash
make test
make integration-tests
make lint
make format
```

Integration tests are skipped unless `ANTHROPIC_API_KEY` is set.

## Deploy to LangSmith

1. Push this template to a Git repository.
2. In LangSmith, create a new Deployment from that repo.
3. Set required environment variables (`ANTHROPIC_API_KEY`, optionally `LANGSMITH_API_KEY`).
4. Deploy using `langgraph.json` defaults.

## Reference docs

- LangChain quickstart: https://docs.langchain.com/oss/python/langchain/quickstart
- LangChain deployment: https://docs.langchain.com/oss/python/langchain/deploy
