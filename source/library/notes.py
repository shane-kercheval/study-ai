"""Classes for creating and managing notes."""

from abc import ABC, abstractmethod
from copy import deepcopy
from functools import cache
from textwrap import dedent
from pydantic import BaseModel
from enum import Enum
import numpy as np
import hashlib
from dataclasses import dataclass

from source.library.helpers import softmax_dict


class Priority(str, Enum):
    """Priority of the note, which is used to determine how often to study the note."""

    low = 'low'
    medium = 'medium'
    high = 'high'


class SubjectMetadata(BaseModel):
    """Metadata about the subject of the note."""

    name: str
    ident: str | None = None
    category: str | None = None
    abbreviation: str | None = None


class NoteMetadata(BaseModel):
    """Metadata about the note."""

    source_name: str
    source_reference: str | None = None  # e.g. url or book information
    reference: str | None = None  # e.g. url or chapeter/page number
    tags: list[str] = []


class Note(ABC):
    """Abstract class that represents a note."""

    def __init__(
            self,
            subject_metadata: SubjectMetadata,
            note_metadata: NoteMetadata,
            priority: Priority = Priority.medium,
            ):
        """Initialize the note with metadata."""
        self.subject_metadata = subject_metadata
        self.note_metadata = note_metadata
        if not isinstance(priority, Priority):
            raise ValueError(f"priority must be a Priority enum: {priority}")
        self.priority = priority

    @cache
    def uuid(self) -> str:
        """Return a unique identifier for the class (e.g. a hash of the content)."""
        subject_meta_content = '-'.join([f"{k}={v}" for k, v in dict(self.subject_metadata).items()])  # noqa
        note_meta_content = '-'.join([f"{k}={v}" for k, v in dict(self.note_metadata).items()])
        text = subject_meta_content + note_meta_content + self.text()
        return hashlib.sha256(text.encode()).hexdigest()

    @abstractmethod
    def text(self) -> str:
        """Render all of the text of the note."""


class TextNote(Note):
    """A TextNote is a Note that has only text."""

    def __init__(
            self,
            text: str,
            subject_metadata: SubjectMetadata,
            note_metadata: NoteMetadata,
            priority: Priority = Priority.medium,
            ):
        super().__init__(
            subject_metadata=subject_metadata,
            note_metadata=note_metadata,
            priority=priority,
        )
        self._text = dedent(text).strip()

    def text(self) -> str:
        """Return the text of the note."""
        return self._text


class Flashcard(Note):
    """A Flashcard is an abstract Note that has a 'preview' and a 'answer'."""

    @abstractmethod
    def preview(self) -> str:
        """Render the 'preview' e.g. the 'question' or 'term' to give context."""

    @abstractmethod
    def answer(self) -> str:
        """Render the 'answer' or 'definition' of the flashcard."""


class DefinitionNote(Flashcard):
    """A DefinitionNote is a Flashcard that has a term and a definition."""

    def __init__(
            self,
            term: str, definition: str,
            subject_metadata: SubjectMetadata,
            note_metadata: NoteMetadata,
            priority: Priority = Priority.medium,
            ):
        super().__init__(
            subject_metadata=subject_metadata,
            note_metadata=note_metadata,
            priority=priority,
        )
        self._term = dedent(term).strip()
        self._definition = dedent(definition).strip()

    def preview(self) -> str:
        """Return the term as the preview."""
        return self._term

    def answer(self) -> str:
        """Return the definition as the answer."""
        return self._definition

    def text(self) -> str:
        """Return the term and definition together to represent the full text."""
        return self._term + " " + self._definition


class QuestionAnswerNote(Flashcard):
    """A QuestionAnswerNote is a Flashcard that has a question and an answer."""

    def __init__(
            self,
            question: str, answer: str,
            subject_metadata: SubjectMetadata, note_metadata: NoteMetadata,
            priority: Priority = Priority.medium,
            ):
        super().__init__(
            subject_metadata=subject_metadata,
            note_metadata=note_metadata,
            priority=priority,
        )
        self._question = dedent(question).strip()
        self._answer = dedent(answer).strip()

    def preview(self) -> str:
        """Return the question as the preview."""
        return self._question

    def answer(self) -> str:
        """Return the answer."""
        return self._answer

    def text(self) -> str:
        """Return the question and answer together to represent the full text."""
        return self._question + " " + self._answer


def dict_to_notes(data: dict) -> list[Note]:
    """
    Creates a list of Note objects from a dictionary/yaml.

    An example yaml file:
    ```
    subject_metadata:
      category: OMSCS
      ident: CS 6200
      name: Graduate Introduction to Operating Systems
      abbreviation: GIOS
    note_metadata:
      source_name: Beej's Guide to Network Programming
      source_reference: https://beej.us/guide/bgnet/pdf/bgnet_usl_c_1.pdf
      tags:
        - systems
        - c
        - networking
        - beejs
    notes:
      - term: What is a `socket`?
        definition: A **way to speak to other programs** using standard Unix **file descriptors**.
        reference: https://beej.us/guide/bgnet/html/#what-is-a-socket; chapter 2
      - term: What is a `file descriptor`?
        definition: An **integer** associated with an **open file**.
        reference: https://beej.us/guide/bgnet/html/#what-is-a-socket; chapter 2
    ```
    """
    data = deepcopy(data)  # don't modify the original data
    notes = []
    for note_dict in data['notes']:
        reference = note_dict.pop('reference', None)
        priority = note_dict.pop('priority', 'medium')
        priority = Priority[priority]
        subject_metadata = SubjectMetadata(**data['subject_metadata'])
        note_metadata = NoteMetadata(
            **data['note_metadata'] | {'reference': reference},
        )
        if isinstance(note_dict, str):
            note = TextNote(text=note_dict)
        elif isinstance(note_dict, dict):
            if 'text' in note_dict:
                note = TextNote(
                    **note_dict,
                    subject_metadata=subject_metadata,
                    note_metadata=note_metadata,
                    priority=priority,
                )
            elif 'term' in note_dict and 'definition' in note_dict:
                note = DefinitionNote(
                    **note_dict,
                    subject_metadata=subject_metadata,
                    note_metadata=note_metadata,
                    priority=priority,
                )
            elif 'question' in note_dict and 'answer' in note_dict:
                note = QuestionAnswerNote(
                    **note_dict,
                    subject_metadata=subject_metadata,
                    note_metadata=note_metadata,
                    priority=priority,
                )
            else:
                raise ValueError(f"Invalid note type: {note_dict}")
        else:
            raise ValueError(f"Invalid note type: {note_dict}")
        notes.append(note)
    return notes


@dataclass
class History:
    """
    Represents past quizes of a note, which includes the number of times the note has been answered
    correctly and incorrectly. This information is used to determine the probability of drawing the
    note in the future.

    The probability of drawing the note is based on the beta distribution, which is a distribution
    over the interval [0, 1]. The beta distribution is basically a distribution over probabilities.
    The more times the note has been answered correctly, the higher the probability that the user
    will answer the question correctly in the future, so the less likely we need to study this
    note.
    """

    correct: int = 0
    incorrect: int = 0

    def probability_correct(self, seed: int | None = None) -> float:
        """
        Draw a random sample from the beta distribution. The interpretation of the value returned
        is the probability of "success" (in this case successfully answering the question
        correctly). The higher the likelihood of success, the less likely we need to study this
        note.

        With no history (0 correct, 0 incorrect; alpha=1, beta=1), the distribution is uniform. As
        the number of correct answers increases, the distribution shifts to the right (higher
        probability of success). As the number of incorrect answers increases, the distribution
        shifts to the left (lower probability of success). As the number of correct and incorrect
        answers increase, the distribution becomes more peaked around the true probability of
        success.
        """
        rng = np.random.default_rng(seed)
        return rng.beta(self.correct + 1, self.incorrect + 1, 1)[0]

    def answer(self, correct: bool) -> None:
        """Update the history based on the correctness of the answer."""
        if correct:
            self.correct += 1
        else:
            self.incorrect += 1

    def to_dict(self) -> dict:
        """Return the history as a dictionary."""
        return {
            'correct': self.correct,
            'incorrect': self.incorrect,
        }


class NoteBank:
    """Represents a collection of notes that can be drawn from to create a quiz."""

    def __init__(
            self,
            notes: list[Note],
            history: dict[str, History] | None = None,
        ):
        """
        Initialize the test bank with notes and history.

        Args:
            notes:
                A list of notes to be included in the test bank.
            history:
                A dictionary of History objects for each note. The key is the UUID of the note
                and the value is the History object.
        """
        self.notes = {}
        for note in notes:
            uuid = note.uuid()
            assert uuid not in self.notes, f"Duplicate UUID: {uuid}"
            self.notes[uuid] = {
                'uuid': uuid,
                'history': history[uuid] if history and uuid in history else History(),
                'note': note,
            }

    def __len__(self) -> int:
        """Return the number of notes in the test bank."""
        return len(self.notes)

    def draw(
            self,
            seed: int | None = None,
            priority_weights: dict[Priority, float] | None = None,
            ) -> Note:
        """
        Draw a note from the test bank. The probability of drawing a note is based on the history
        of the note. The more times the note has been answered correctly, the less likely we need
        to study this note.

        Returns a dictionary with the UUID of the note, the history of the note, and the note
        itself.

        # TODO: implement priority_weights.
        """
        probabilities = {}
        # probability_correct gives the probability of success (correct answer), but the higher
        # the probability of success, the less likely we need to study this note, so we
        # subtract the value from 1 to get the probability of incorrectly answering the
        # question. The higher the probability of incorrectly answering the question, the more
        # likely we need to study this note.
        if priority_weights:
            priority_weights = softmax_dict(priority_weights)
            for k, v in self.notes.items():
                probability_incorrect = 1 - v['history'].probability_correct()
                probabilities[k] = probability_incorrect * priority_weights[ v['note'].priority]
        else:
            probabilities = {
                k: 1 - v['history'].probability_correct()
                for k, v in self.notes.items()
            }
        probabilities = softmax_dict(probabilities)
        assert np.isclose(sum(probabilities.values()), 1), f"Invalid probabilities: {probabilities}"  # noqa
        # draw a note
        rng = np.random.default_rng(seed)
        uuid = rng.choice(list(probabilities.keys()), p=list(probabilities.values()))
        return self.notes[uuid]['note']

    def answer(self, uuid: str, correct: bool) -> None:
        """Update the history of the note based on the correctness of the answer."""
        self.notes[uuid]['history'].answer(correct)

    def history(self, to_dict: bool = False) -> dict[str, History] | dict[str, dict[str, int]]:
        """
        Return the history of the notes.

        Args:
            to_dict:
                If True, return the history as a dictionary of dictionaries. If False, return the
                history as a dictionary of History objects.
        """
        if to_dict:
            return {uuid: v['history'].to_dict() for uuid, v in self.notes.items()}
        return {uuid: v['history'] for uuid, v in self.notes.items()}
