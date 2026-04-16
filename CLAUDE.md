# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Jaxon Core is the brain and memory system for Jaxon Vale, an AI partner project. The stack is Python with Flask (web server) and Requests (HTTP client). The project is in early scaffolding — application modules do not yet exist.

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Development Commands

As the project grows, commands should follow this pattern (update this file when tooling is added):

```bash
# Run the Flask app (once an app entry point exists)
flask run
# or
python app.py

# Run tests (once pytest is added to requirements.txt)
python -m pytest
python -m pytest tests/test_foo.py::test_bar   # single test

# Lint (once a linter is configured)
flake8 .
black .
```

## Architecture

The project is structured around two core concerns:

- **Brain** — reasoning/decision logic for Jaxon's responses
- **Memory** — persistent storage and retrieval of context/history

Flask serves as the HTTP interface. `requests` is used for outbound calls to external services (e.g., LLM APIs, integrations).

When adding modules, keep brain and memory concerns separated. Entry points (Flask routes) should stay thin — delegate logic to brain/memory modules.

## Conventions

- Python only; no JavaScript/TypeScript
- Dependencies go in `requirements.txt` (pin versions when adding)
- Branch off `main`; the active development branch is `claude/add-claude-documentation-DLA9i`
