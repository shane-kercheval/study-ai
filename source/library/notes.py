# load yaml file

from abc import ABC, abstractmethod
from textwrap import dedent
from pydantic import BaseModel, field_validator
from enum import Enum
import numpy as np
import hashlib
from dataclasses import dataclass


class Priority(str, Enum):
    low = 'low'
    medium = 'medium'
    high = 'high'


class Note(ABC, BaseModel):
    priority: Priority = Priority.medium
    reference: str | None = None
    tags: list[str] = []

    def uuid(self) -> str:
        """Return a unique identifier for the note (e.g. a hash of the content)."""
        content = '-'.join([f"{k}={v}" for k, v in self.dict().items()])
        return hashlib.sha256(content.encode()).hexdigest()

    @abstractmethod
    def note(self):
        """Render the 'description' e.g. the 'answer' or 'definition' to provide more detail."""

class Flashcard(Note):
    @abstractmethod
    def preview(self):
        """Render the 'preview' e.g. the 'question' or 'term' to give context."""

class TextNote(Note):
    text: str

    @field_validator('text')
    @classmethod
    def text_validator(cls, t: str) -> str:
        return dedent(t).strip()

    def preview(self):
        return self.text

    def note(self):
        return self.text


class DefinitionNote(Flashcard):
    term: str
    definition: str

    @field_validator('term')
    @classmethod
    def term_validator(cls, t: str) -> str:
        return dedent(t).strip()
    
    @field_validator('definition')
    @classmethod
    def definition_validator(cls, d: str) -> str:
        return dedent(d).strip()

    def preview(self):
        return self.term

    def note(self):
        return self.term + self.definition


class QuestionAnswerNote(Flashcard):
    question: str
    answer: str

    @field_validator('question')
    @classmethod
    def term_validator(cls, q: str) -> str:
        return dedent(q).strip()
    
    @field_validator('answer')
    @classmethod
    def definition_validator(cls, a: str) -> str:
        return dedent(a).strip()

    def preview(self):
        return self.question

    def note(self):
        return self.question + self.answer


class ClassNote(BaseModel):

    subject_metadata: dict = {}
    note_metadata: dict = {}
    note: Note

    def uuid(self) -> str:
        """Return a unique identifier for the class (e.g. a hash of the content)."""
        subject_meta_content = '-'.join([f"{k}={v}" for k, v in self.subject_metadata.items()])
        note_meta_content = '-'.join([f"{k}={v}" for k, v in self.note_metadata.items()])
        return hashlib.sha256((subject_meta_content + note_meta_content).encode()).hexdigest() + note.uuid()  # noqa

def parse(data: dict) -> list[ClassNote]:
    """Create a list of ClassNote objects from a dictionary."""
    notes = []
    for note_dict in data['notes']:
        if isinstance(note_dict, str):
            note = TextNote(text=note_dict)
        elif isinstance(note_dict, dict):
            if 'text' in note_dict:
                note = TextNote(**note_dict)
            elif 'term' in note_dict and 'definition' in note_dict:
                note = DefinitionNote(**note_dict)
            elif 'question' in note_dict and 'answer' in note_dict:
                note = QuestionAnswerNote(**note_dict)
            else:
                raise ValueError(f"Invalid note type: {note_dict}")
        else:
            raise ValueError(f"Invalid note type: {note_dict}")
        class_note = ClassNote(
            subject_metadata=data['subject_metadata'],
            note_metadata=data['note_metadata'],
            note=note,
        )
        notes.append(class_note)
    return notes



@dataclass
class History:
    correct: int = 0
    incorrect: int = 0

    def beta_draw(self):
        """
        Draw a sample from the beta distribution. The interpretation is the probability of
        "success" (in this case successfully answering the question correctly). The higher the
        likelihood of success, the less likely we need to study this note.
        """
        return np.random.beta(self.correct + 1, self.incorrect + 1, 1)[0]

    def correct_answer(self, correct: bool) -> None:
        if correct:
            self.correct += 1
        else:
            self.incorrect += 1

    def to_dict(self) -> dict:
        return {
            'correct': self.correct,
            'incorrect': self.incorrect,
        }


class TestBank:
    def __init__(
            self,
            notes: list[ClassNote],
            history: dict[str, History] | None = None,
            flash_only: bool = False,
            class_category: str | None = None,
            class_ident: str | None = None,
            class_name: str | None = None,
            class_abbr: str | None = None,
        ):
        """TODO."""
        self.test_bank = {}
        class_attributes = {
            'category': class_category,
            'ident': class_ident,
            'name': class_name,
            'abbr': class_abbr
        }

        for class_note in notes:
            for note in class_note.notes:
                if flash_only and not isinstance(note, Flashcard):
                    continue
                if any(
                    class_note.subject_metadata.get(attr) != value
                    for attr, value in class_attributes.items() if value
                    ):
                    continue
                uuid = class_note.uuid() + note.uuid()
                assert uuid not in self.test_bank, f"Duplicate UUID: {uuid}"
                self.test_bank[uuid] = {
                    'uuid': uuid,
                    'history': history[uuid] if history and uuid in history else History(),
                    'note': note,
                }

    def draw(self) -> dict:
        """Draw a note from the class notes."""
        probabilities = {
            k: v['history'].beta_draw()
            for k, v in self.test_bank.items()
        }
        # softmax probabilities across all values
        sum_probs = sum(probabilities.values())
        probabilities = {k: v / sum_probs for k, v in probabilities.items()}
        # draw a note
        uuid = np.random.choice(list(probabilities.keys()), p=list(probabilities.values()))
        return self.test_bank[uuid]

    def correct_answer(self, uuid: str, correct: bool) -> None:
        """Update the history of the note based on the correctness of the answer."""
        self.test_bank[uuid]['history'].correct_answer(correct)

    @property
    def history(self) -> dict[str, History]:
        return {k: v['history'] for k, v in self.test_bank.items()}
