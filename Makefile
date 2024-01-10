.PHONY = setup format lint static test

all: ${.PHONY}

test:
	pytest --cov=. --cov-report term-missing --cov-fail-under=70 -vvl --numprocesses auto

static:
	mypy ml.py ml_test.py

lint:
	pylint ml.py ml_test.py --fail-under 8

format:
	black ml.py ml_test.py

setup:
	pip install -r requirements.txt