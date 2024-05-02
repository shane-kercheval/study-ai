# study-ai
Using Generative AI to aid in studying

## Features

- Lightweight terminal interface
- Notes:
    - Cycle through notes or flash cards sequentially
- Flash Cards
    - uses probabilities (based on number of correct answers) to randomly pick notes
- Search
    - uses embeddings and vector database (i.e. numpy matrix and cosine similarity)
- Generate Flashcards with AI (coming soon)
- Chat with AI using notes (coming soon)
    - e.g. Summarize key points from notes
- AI Quizes (coming soon)


- Command line tool for extract text from pdf

## Downsides / Limitations

- notes need to be in yaml format
- yaml requires specific formatting (it will suck if you take a lot of notes and then have a weird formatting error somewhere preventing your notes from loading)
- images are not supported
- terminal interface means it's not supported on phone/tablet


# Using

- docker
- probably want to fork so you can check in your own history (i.e. number of questions answered correctly)

# Contributing

- **Do not submit course information (e.g. information/answers for projects/quizes/exams) that violates class/school policy.**


- unit tests




# Requirements

* Extract text from PDF
* Agent to create flashcards from text (notes, text from PDF)
    * Ensure structure (e.g. OpenAI functions?)
        * Possibly iterate if structure is incorrect (can’t load dict)
    * Possibly have a way to get feedback and generate more flashcards
    * Way to evaluate creation of flashcards
        * Quality/missing topics (via llm)
        * Structure (e.g. dictionary/yaml)
    * Should have `sources` e.g. notes, text from pdf
    * Output should be yaml of flashcards.
    * Perhaps there should be a directory of flashcards where user can choose topic or choose all
* Multiple choice vs “grading” and feedback
    * Not sure I like multiple choice
* Option like Anki where I can indicate whether or not I got the answer correct, which allows the program to calculate if it should display
        * Should not affect “time” like I think Anki does but should affect the probability distribution
            * e.g. each card has e.g. last 5 responses and 0/5 correct gets a higher probability of 3/5 correct > 5/5 correct.
    *


* Flash cards app is separate from agents to create flash cards; step 1 is get simple flashcard app

