.PHONY = setup format lint static test

all: ${.PHONY}

test:
	python -m pytest --cov=. --cov-report term-missing --cov-fail-under=70 -vvl --numprocesses auto

static:
	python -m mypy ml.py ml_test.py

lint:
	python -m pylint ml.py ml_test.py --fail-under 8

format:
	python -m black ml.py ml_test.py

setup:
	pip install -r requirements.txt
