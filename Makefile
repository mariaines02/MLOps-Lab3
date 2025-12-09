install:
	pip install uv &&\
	uv sync

test:
	uv run python -m pytest ./tests -vv  --cov=logic --cov=api --cov=cli 

format:	
	uv run black logic/*.py api/*.py cli/*.py src/*.py

lint:
	uv run pylint --rcfile=.pylintrc --ignore-patterns=test_.*\.py logic/*.py api/*.py cli/*.py src/*.py

refactor: format lint
		
all: install format lint test
