####
# Commands that I use to start the docker container and run the study.py CLI.
# The directories point to the notes that I have taken for the OMSCS program, which are stored in a
# separate/private repository which is mounted to the container in the `/notes` directory.
####
start:
	docker-compose run -v /Users/shanekercheval/repos/omscs:/notes bash /bin/bash

image-reduction:
	uv run python image_reduction.py

clean_markdown_notes:
	uv run python study.py format-notes \
		--model_type openai \
		--model_name gpt-4o-mini \
	    --temperature 0.5 \
		--stream \
		--clean_markdown \
		--file ../obsidian_omscs/CS6200-GIOS/Module\ Notes/P2L5\ -\ Thread\ Performance.md

study_gios:
	uv run python study.py study \
		--notes_paths "../omscs/CS-6200-GIOS/flash-cards/*.yaml" \
		--history_path "../omscs/CS-6200-GIOS/flash-cards/study-ai/history.yaml"

smart_study_gios:
	uv run python study.py study \
		--smart_cycle \
		--notes_paths "../omscs/CS-6200-GIOS/flash-cards/*.yaml" \
		--history_path "../omscs/CS-6200-GIOS/flash-cards/study-ai/history.yaml"

search:
	uv run python study.py search \
		--notes_paths "../omscs/CS-6200-GIOS/flash-cards/*.yaml" \
		--db_path "../omscs/CS-6200-GIOS/flash-cards/study-ai/vector_db.parquet" \
		--top_k 5 \
		--similarity_threshold 0.3

text_to_flashcards:
	uv run python study.py text-to-flashcards \
		--file ../obsidian_omscs/CS6200-GIOS/Module\ Notes/P1L2\ -\ 2\ -\ Intro\ to\ OS.md

quiz:
	uv run python study.py quiz \
	    --temperature 1 \
		--file ../obsidian_omscs/CS6200-GIOS/Module\ Notes/P1L2\ -\ 1\ -\ Intro\ to\ OS.md

quiz_local:
	uv run python study.py quiz \
		--model_type openai_server \
		--model_name http://localhost:1234/v1 \
		--stream \
	    --temperature 1 \
		--file ../obsidian_omscs/CS6200-GIOS/Module\ Notes/P1L2\ -\ 1\ -\ Intro\ to\ OS.md

format_to_markdown:
	uv run python study.py format-notes \
		--model_type openai \
		--model_name gpt-4o-mini \
	    --temperature 0.5 \
		--stream

format_to_markdown_local:
	uv run python study.py format-notes \
		--model_type openai_server \
		--model_name http://localhost:1234/v1 \
	    --temperature 0.5 \
		--stream

####
# CLI Examples
####
pdf_to_text:
	pdftotext /notes/CS-6200-GIOS/pdfs/OSTEP-Chapter-2.pdf /notes/CS-6200-GIOS/pdfs/OSTEP-Chapter-2.txt

study_docker:
	# launch study.py in docker container 
	docker exec -it study-ai-bash-1 /bin/zsh -c "python study.py study"

study_default:
	# start `study` with default settings/directory 
	python study.py study

study_abbreviation_gios:
	# start `study` filtering for only GIOS class abbreviation
	python study.py study --a GIOS

study_flashcards:
	# start `study` with flashcards only
	python study.py study --flash_only

search_default:
	# start `search`
	python study.py search

text_to_flashcards_default:
	python study.py text-to-flashcards

text_to_flashcards_local:
	# using lmstudio to set up local server
	python study.py text-to-flashcards \
		--model_type openai_server \
		--model_name http://host.docker.internal:1234/v1

text_to_flashcards_file:
	# using lmstudio to set up local server
	python study.py text-to-flashcards \
		--model_name gpt-4o-mini \
		--file /code/temp.txt
		# --model_type openai_server \
		# --model_name http://host.docker.internal:1234/v1 \
####
# DOCKER (NOTE i moved from using docker to using conda)
####
# docker_build:
# 	docker compose -f docker-compose.yml build

# docker_run: docker_build
# 	docker compose -f docker-compose.yml up

# docker_down:
# 	docker compose down --remove-orphans

####
# Project
####
linting:
	uv run ruff check source/library
	uv run ruff check source/cli
	uv run ruff check study.py
	uv run ruff check tests

unittests:
	rm -f tests/test_files/log.log
	# pytest tests
	uv run coverage run -m pytest --durations=0 tests
	uv run coverage html

# doctests:
# 	python -m doctest source/library/utilities.py

tests: linting unittests
