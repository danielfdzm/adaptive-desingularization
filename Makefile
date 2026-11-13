PYTHON ?= python3

.DEFAULT_GOAL := help
.PHONY: help check list figures tables

help:
	@$(PYTHON) reproduce.py --help

check:
	$(PYTHON) reproduce.py check

list:
	$(PYTHON) reproduce.py list

figures:
	$(PYTHON) reproduce.py figures

tables:
	$(PYTHON) reproduce.py tables
