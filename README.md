# study-ai
Using Generative AI to aid in studying


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

