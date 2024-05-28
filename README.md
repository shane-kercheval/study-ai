# study-ai

Lightweight terminal interface using Generative AI and semantic search to aid in studying.

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

- docker
- probably want to fork so you can check in your own history (i.e. number of questions answered correctly) and local vector database.

# Contributing

- **Do not submit course information (e.g. information/answers for projects/quizes/exams) that violates class/school policy.**
- please write unit tests following correct patterns
