####
# CLI
####
study:
	docker exec -it study-ai-bash-1 /bin/zsh -c "python study.py cycle"

####
# DOCKER
####
docker_build:
	docker compose -f docker-compose.yml build

docker_run: docker_build
	docker compose -f docker-compose.yml up

docker_down:
	docker compose down --remove-orphans

####
# Project
####
linting:
	ruff check source/config
	ruff check source/entrypoints
	ruff check source/library
	ruff check source/notebooks
	ruff check source/service
	ruff check tests

unittests:
	rm -f tests/test_files/log.log
	# pytest tests
	coverage run -m pytest --durations=0 tests
	coverage html

doctests:
	python -m doctest source/library/utilities.py

tests: linting unittests doctests

remove_logs:
	rm -f output/log.log

## Run entire workflow.
all: tests remove_logs data explore

## Delete all generated files (e.g. virtual)
clean:
	rm -f data/raw/*.pkl
	rm -f data/raw/*.csv
	rm -f data/processed/*
