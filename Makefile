UV        ?= uv

MYPY_FLAGS = --warn-return-any --warn-unused-ignores \
             --ignore-missing-imports --disallow-untyped-defs \
             --check-untyped-defs

.PHONY: install run debug clean lint lint-strict

install:
	$(UV) sync

run:
	$(UV) run python -m src

debug:
	$(UV) run python3 -m pdb -m src

lint:
	$(UV) run flake8 src
	$(UV) run mypy src $(MYPY_FLAGS)

lint-strict:
	$(UV) run flake8 src
	$(UV) run mypy src --strict

clean:
	rm -rf __pycache__ .mypy_cache .pytest_cache
	find . -name '__pycache__' -type d -prune -exec rm -rf {} +
	find . -name '*.pyc' -delete