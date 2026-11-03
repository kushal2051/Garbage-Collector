.PHONY: setup lint format test clean

UV := $(shell command -v uv 2>/dev/null)

setup:  ## create .venv and install dev dependencies
ifdef UV
	uv sync --dev
else
	python3 -m venv .venv && .venv/bin/pip install -e . pytest ruff
endif

lint:
	uv run ruff check . && uv run ruff format --check .

format:
	uv run ruff check --fix . && uv run ruff format .

test: lint
	uv run pytest -q

skills:  ## refresh the .claude/skills symlinks from .agents/skills
	python3 scripts/sync_claude_skills.py --prune

clean:
	rm -rf .venv .pytest_cache .ruff_cache
	find . -name __pycache__ -type d -exec rm -rf {} +
