.PHONY: install run debug clean lint lint-strict

install:
	uv sync

run:
	HF_TOKEN=$(HF_TOKEN) uv run python -m src

debug:
	time uv run python -m src

clean:
	rm -rf */__pycache__ */.mypy_cache data/output/
	uv cache clean

lint:
	uv run flake8 .
	uv run mypy . --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

lint-strict:
	uv run flake8 .
	uv run mypy . --strict