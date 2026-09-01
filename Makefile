.PHONY: install test lint format typecheck benchmark dashboard clean

install:
	pip install -e ".[dev,dashboard]"

test:
	pytest -v

lint:
	ruff check .

format:
	ruff check . --fix
	ruff format .

typecheck:
	mypy src/mem0_bench

benchmark:
	mem0-bench generate --users 10 --turns 50 --seed 42
	mem0-bench compare --backends mem0,rag,full-context \
		--dataset datasets/conversations/synthetic_u10_t50_s42.json

dashboard:
	mem0-bench dashboard

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache .mypy_cache .ruff_cache .coverage htmlcov
