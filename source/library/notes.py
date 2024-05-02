# load yaml file

from abc import ABC, abstractmethod
from textwrap import dedent
from pydantic import BaseModel, field_validator
import yaml
from enum import Enum

# enum Priority
class Priority(str, Enum):
    low = 'low'
    medium = 'medium'
    high = 'high'


class Note(ABC, BaseModel):
    priority: Priority = Priority.medium
    reference: str | None = None
    tags: list[str] = []

    def uuid(self):
        """Return a unique identifier for the note (e.g. a hash of the content)."""
        return hash('-'.join([f"{k}={v}" for k, v in self.dict().items()]))

    @abstractmethod
    def preview(self):
        """Render the 'preview' e.g. the 'question' or 'term' to give context."""

    @abstractmethod
    def note(self):
        """Render the 'description' e.g. the 'answer' or 'definition' to provide more detail."""


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


class DefinitionNote(Note):
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
        return self.definition


class QuestionAnswerNote(Note):
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
        return self.answer  


class ClassNotes(BaseModel):
    name: str
    notes: list[Note] = []
    ident: str | None = None
    abbreviation: str | None = None
    category: str | None = None
    tags: list[str] = []

    @classmethod
    def from_dict(cls, data: dict):
        """Create a ClassNotes object from a dictionary."""
        notes = []
        for note in data['notes']:
            if isinstance(note, str):
                notes.append(TextNote(text=note))
            elif isinstance(note, dict):
                if 'text' in note:
                    notes.append(TextNote(**note))
                elif 'term' in note and 'definition' in note:
                    notes.append(DefinitionNote(**note))
                elif 'question' in note and 'answer' in note:
                    notes.append(QuestionAnswerNote(**note))
                else:
                    raise ValueError(f"Invalid note type: {note}")
            else:
                raise ValueError(f"Invalid note type: {note}")
        data['notes'] = notes
        return ClassNotes(**data)


with open("../../tests/test_files/notes1.yaml", "r") as f:
    notes = yaml.safe_load(f)


class_notes = ClassNotes.from_dict(notes)
print(class_notes)