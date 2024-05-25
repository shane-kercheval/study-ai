"""Classes for creating and managing notes."""

from abc import ABC, abstractmethod
from copy import deepcopy
from textwrap import dedent
from pydantic import BaseModel
from enum import Enum
import numpy as np
import hashlib
from dataclasses import dataclass


class Priority(str, Enum):
    """Priority of the note, which is used to determine how often to study the note."""

    low = 'low'
    medium = 'medium'
    high = 'high'


class SubjectMetadata(BaseModel):
    """Metadata about the subject of the note."""

    category: str
    ident: str
    name: str
    abbreviation: str


class NoteMetadata(BaseModel):
    """Metadata about the note."""

    source_name: str
    source_reference: str  # e.g. url or book information
    reference: str | None = None  # e.g. url or chapeter/page number
    priority: Priority = Priority.medium
    tags: list[str] = []


class Note(ABC):
    """Abstract class that represents a note."""

    def __init__(self, subject_metadata: SubjectMetadata, note_metadata: NoteMetadata):
        """Initialize the note with metadata."""
        self.subject_metadata = subject_metadata
        self.note_metadata = note_metadata

    def uuid(self) -> str:
        """Return a unique identifier for the class (e.g. a hash of the content)."""
        subject_meta_content = '-'.join([f"{k}={v}" for k, v in self.subject_metadata.items()])
        note_meta_content = '-'.join([f"{k}={v}" for k, v in self.note_metadata.items()])
        text = subject_meta_content + note_meta_content + self.text()
        return hashlib.sha256(text.encode()).hexdigest()

    @abstractmethod
    def text(self) -> str:
        """Render all of the text of the note."""


class TextNote(Note):
    """A TextNote is a Note that has only text."""

    def __init__(self, subject_metadata: SubjectMetadata, note_metadata: NoteMetadata, text: str):
        super().__init__(subject_metadata=subject_metadata, note_metadata=note_metadata)
        self._text = dedent(text).strip()

    def text(self) -> str:
        """Return the text of the note."""
        return self._text


class Flashcard(Note):
    """A Flashcard is a Note that has a 'preview' and a 'answer'."""

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
            subject_metadata: SubjectMetadata, note_metadata: NoteMetadata,
            term: str, definition: str):
        super().__init__(subject_metadata=subject_metadata, note_metadata=note_metadata)
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
        return self._term + self._definition


class QuestionAnswerNote(Flashcard):
    """A QuestionAnswerNote is a Flashcard that has a question and an answer."""

    def __init__(
            self,
            subject_metadata: SubjectMetadata, note_metadata: NoteMetadata,
            question: str, answer: str):
        super().__init__(subject_metadata=subject_metadata, note_metadata=note_metadata)
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
        return self._question + self._answer


def parse(data: dict) -> list[Note]:
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
        priority = note_dict.pop('priority', Priority.medium)
        subject_metadata = SubjectMetadata(**data['subject_metadata'])
        note_metadata = NoteMetadata(
            **data['note_metadata'] | {'reference': reference, 'priority': priority},
        )
        if isinstance(note_dict, str):
            note = TextNote(text=note_dict)
        elif isinstance(note_dict, dict):
            if 'text' in note_dict:
                note = TextNote(
                    **note_dict,
                    subject_metadata=subject_metadata,
                    note_metadata=note_metadata,
                )
            elif 'term' in note_dict and 'definition' in note_dict:
                note = DefinitionNote(
                    **note_dict,
                    subject_metadata=subject_metadata,
                    note_metadata=note_metadata,
                )
            elif 'question' in note_dict and 'answer' in note_dict:
                note = QuestionAnswerNote(
                    **note_dict,
                    subject_metadata=subject_metadata,
                    note_metadata=note_metadata,
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

    def beta_draw(self, seed: int | None = None) -> float:
        """
        Draw a sample from the beta distribution. The interpretation is the probability of
        "success" (in this case successfully answering the question correctly). The higher the
        likelihood of success, the less likely we need to study this note.
        """
        rng = np.random.default_rng(seed)
        return rng.random.beta(self.correct + 1, self.incorrect + 1, 1)[0]

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


# class TestBank:
#     def __init__(
#             self,
#             notes: list[SubjectNote],
#             history: dict[str, History] | None = None,
#             flash_only: bool = False,
#             class_category: str | None = None,
#             class_ident: str | None = None,
#             class_name: str | None = None,
#             class_abbr: str | None = None,
#         ):
#         """TODO."""
#         self.test_bank = {}
#         class_attributes = {
#             'category': class_category,
#             'ident': class_ident,
#             'name': class_name,
#             'abbr': class_abbr
#         }

#         for class_note in notes:
#             for note in class_note.notes:
#                 if flash_only and not isinstance(note, Flashcard):
#                     continue
#                 if any(
#                     class_note.subject_metadata.get(attr) != value
#                     for attr, value in class_attributes.items() if value
#                     ):
#                     continue
#                 uuid = class_note.uuid() + note.uuid()
#                 assert uuid not in self.test_bank, f"Duplicate UUID: {uuid}"
#                 self.test_bank[uuid] = {
#                     'uuid': uuid,
#                     'history': history[uuid] if history and uuid in history else History(),
#                     'note': note,
#                 }

#     def draw(self) -> dict:
#         """Draw a note from the class notes."""
#         probabilities = {
#             k: v['history'].beta_draw()
#             for k, v in self.test_bank.items()
#         }
#         # softmax probabilities across all values
#         sum_probs = sum(probabilities.values())
#         probabilities = {k: v / sum_probs for k, v in probabilities.items()}
#         # draw a note
#         uuid = np.random.choice(list(probabilities.keys()), p=list(probabilities.values()))
#         return self.test_bank[uuid]

#     def correct_answer(self, uuid: str, correct: bool) -> None:
#         """Update the history of the note based on the correctness of the answer."""
#         self.test_bank[uuid]['history'].correct_answer(correct)

#     @property
#     def history(self) -> dict[str, History]:
#         return {k: v['history'] for k, v in self.test_bank.items()}
