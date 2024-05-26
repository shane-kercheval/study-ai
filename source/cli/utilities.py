"""Contains helper functions that are used in the main program."""
import glob
import re

import yaml

from source.library.notes import Flashcard, History, Note, dict_to_notes


def load_notes(path: str) -> list[Note]:
    """
    Load notes from multiple yaml files.

    Args:
        path:
            The path to the yaml files. The expected format is the same as the glob module.
    """
    class_notes = []
    # load all yaml files in data/notes via glob
    for f in glob.glob(path):
        with open(f) as handle:
            data = yaml.safe_load(handle)
        class_notes.extend(dict_to_notes(data))
    return class_notes


def load_history(file_path: str) -> dict:
    """Load history from a yaml file."""
    with open(file_path) as h:
        history = yaml.safe_load(h)
        return {uuid: History(**history[uuid]) for uuid in history}


def filter_notes(
        notes: list[Note],
        flash_only: bool = False,
        category: str | None = None,
        ident: str | None = None,
        name: str | None = None,
        abbr: str | None = None,
        ) -> list[Note]:
    """
    Filter notes based on various criteria.

    Args:
        notes:
            The list of notes to filter.
        flash_only:
            If True, only return FlashCard instances.
        category:
            The category (in SubjectMetadata) to filter on.
        ident:
            The identity (in SubjectMetadata) to filter on.
        name:
            The name (in SubjectMetadata) to filter on.
        abbr:
            The abbreviation (in SubjectMetadata) to filter on.
    """
    if flash_only:
        notes = [note for note in notes if isinstance(note, Flashcard)]
    if category:
        notes = [note for note in notes if note.subject_metadata.category == category]
    if ident:
        notes = [note for note in notes if note.subject_metadata.ident == ident]
    if name:
        notes = [note for note in notes if note.subject_metadata.name == name]
    if abbr:
        notes = [note for note in notes if note.subject_metadata.abbreviation == abbr]
    return notes


def colorize_markdown(text: str) -> str:
    """Colorizes text (used as the output in the terminal) based on markdown syntax."""
    # Apply red color for text surrounded by **
    text = re.sub(r'\*\*(.*?)\*\*', r'\033[31m\1\033[0m', text)
    # Apply blue color for text surrounded by `
    return re.sub(r'`(.*?)`', r'\033[34m\1\033[0m', text)


def colorize_gray(text: str) -> str:
    """Colorizes text (used as the output in the terminal) to gray."""
    # Apply gray color to all text
    return f'\033[90m{text}\033[0m'
