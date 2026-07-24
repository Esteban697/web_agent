# Web Agent in LangGraph

Web Agent is a flexible up-to-date implementation in LangGraph capable of browsing the web to find useful information for the user.
It can go to a URL, click and read through the content to find relevant information about the human message delivered.
It is deployable in LangGraph Studio to easily test.

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
