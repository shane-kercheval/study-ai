####
# CLI
####
study_docker:
	docker exec -it study-ai-bash-1 /bin/zsh -c "python study.py cycle"

study:
	python study.py cycle

flash:
	python study.py cycle --flash_only

text_to_notes:
	python study.py text-to-notes

text_to_notes_local:
	# using lmstudio to set up local server
	python study.py text-to-notes \
		--model_type openai_server \
		--model_name http://host.docker.internal:1234/v1

text_to_notes_file:
	# using lmstudio to set up local server
	python study.py text-to-notes \
		--model_name gpt-4-turbo-2024-04-09 \
		--file /code/temp.txt
		# --model_type openai_server \
		# --model_name http://host.docker.internal:1234/v1 \

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
	ruff check source/library
	ruff check tests

unittests:
	rm -f tests/test_files/log.log
	# pytest tests
	coverage run -m pytest --durations=0 tests
	coverage html

# doctests:
# 	python -m doctest source/library/utilities.py

tests: linting unittests
