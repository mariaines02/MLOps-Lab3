install:
	pip install uv &&\
	uv sync

test:
	uv run python -m pytest ./tests -vv  --cov=logic --cov=api --cov=cli 

format:	
	uv run black logic/*.py api/*.py cli/*.py

lint:
	uv run pylint --disable=R,C --ignore-patterns=test_.*\.py logic/*.py api/*.py cli/*.py

refactor: format lint
		
all: install format lint test
