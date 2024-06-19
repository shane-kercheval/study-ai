# study-ai

Lightweight terminal interface using Generative AI and semantic search to aid in studying.

This project is optimized for my study preferences. It's not intended to be generalized to meet the needs of a larger audience. As such, documentation is a little light.

Commands in `study.py` CLI:

- `cycle`:
    - Cycle through notes or flash cards.
    - Track correct/incorrect answers.
    - The more correct answers for a note or flash card, the less it will be shown.
- `search`:
    - uses embeddings and vector database (i.e. numpy matrix and cosine similarity)
- `text-to-notes`:
    - Generate notes from a text/file with AI
- Chat with AI using notes (coming soon)

See `Makefile` for examples of how to use the CLI.

## Downsides / Limitations

- notes need to be in yaml format
- yaml requires specific formatting (it will suck if you take a lot of notes and then have a weird formatting error somewhere preventing your notes from loading)
- images are not supported
- terminal interface means it's not supported on phone/tablet

# Using

- To build/run docker contain, you can use the command `make docker_run`. For development, after I've ran this command, I attach VS Code to the container.
- There are a few examples of note files in the `data`, which allows users to run the CLI with default options to get a feel for how it works. However, my notes are in a private github repo, so I point all files (notes, history, vector db) to that repo and corresponding directories.
- When I run the CLI for studying, I use the command `make start`. This starts the container in the terminal, and also attaches the directory that contains all of my notes (i.e. allows the container to see the folder outside of the container), which points to a private github repo in the specified directory.
- See `Makefile` for examples of how to run the various CLI commands.

# Contributing

- This project is optimized for my study preferences. It's not intended to be generalized to meet the needs of a larger audience. As such, documentation is a little light.
- **Do not submit course information (e.g. information/answers for projects/quizes/exams) that violates class/school policy.**
- please write unit tests following correct patterns
