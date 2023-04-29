#!make
include .env
.PHONY: install format
.ONESHELL:

SHELL = /bin/bash
CONDA_ACTIVATE = source $$(conda info --base)/etc/profile.d/conda.sh ; conda activate

all: install format
install:
	conda env update --prune -p ${ENV_FOLDER} -f environment_pip.yml
	$(CONDA_ACTIVATE) ${ENV_FOLDER}
format:
	$(CONDA_ACTIVATE) ${ENV_FOLDER}
	autoflake -r --in-place --remove-unused-variables --remove-all-unused-imports ivae_scorer
	isort --profile black ivae_scorer notebooks
	black ivae_scorer notebooks
