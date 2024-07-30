####
# Commands that I use to start the docker container and run the study.py CLI.
# The directories point to the notes that I have taken for the OMSCS program, which are stored in a
# separate/private repository which is mounted to the container in the `/notes` directory.
####
start:
	docker-compose run -v /Users/shanekercheval/repos/omscs:/notes bash /bin/bash

study_gios:
	python study.py cycle \
		--notes_paths "/notes/CS-6200-GIOS/notes/*.yaml" \
		--history_path /notes/study-ai/history.yaml

search:
	python study.py search \
		--notes_paths "/notes/CS-6200-GIOS/notes/*.yaml" \
		--db_path /notes/study-ai/vector_db.parquet \
		--top_k 5 \
		--similarity_threshold 0.3


text_to_notes:
	pdftotext /notes/CS-6200-GIOS/pdfs/OSTEP-Chapter-2.pdf /notes/CS-6200-GIOS/pdfs/OSTEP-Chapter-2.txt
	python study.py text-to-notes \
		--file /notes/CS-6200-GIOS/pdfs/OSTEP-Chapter-2.txt

####
# CLI Examples
####

pdf_to_text:
	pdftotext /notes/CS-6200-GIOS/pdfs/OSTEP-Chapter-2.pdf /notes/CS-6200-GIOS/pdfs/OSTEP-Chapter-2.txt

study_docker:
	# launch study.py in docker container 
	docker exec -it study-ai-bash-1 /bin/zsh -c "python study.py cycle"

study_default:
	# start `cycle` with default settings/directory 
	python study.py cycle

study_abbreviation_gios:
	# start `cycle` filtering for only GIOS class abbreviation
	python study.py cycle --a GIOS

study_flashcards:
	# start `cycle` with flashcards only
	python study.py cycle --flash_only

search_default:
	# start `search`
	python study.py search

text_to_notes_default:
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
	ruff check source/cli
	ruff check study.py
	ruff check tests

unittests:
	rm -f tests/test_files/log.log
	# pytest tests
	coverage run -m pytest --durations=0 tests
	coverage html

# doctests:
# 	python -m doctest source/library/utilities.py

tests: linting unittests
