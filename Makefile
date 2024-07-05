.PHONY = setup format lint static test

all: ${.PHONY}

test:
	python3 -m pytest --cov=. --cov-report term-missing --cov-fail-under=70 -vvl --numprocesses auto

static:
	python3 -m mypy ml.py ml_test.py

lint:
	python3 -m pylint ml.py ml_test.py --fail-under 8

format:
	python3 -m black ml.py ml_test.py

setup:
	pip3 install -r requirements.txt
