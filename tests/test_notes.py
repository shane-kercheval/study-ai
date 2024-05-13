"""Test utilities.py."""

from copy import deepcopy
from textwrap import dedent
from source.library.notes import Flashcard, Priority, TextNote, parse


def test__parse(fake_notes):
    original_notes = deepcopy(fake_notes)
    notes = parse(fake_notes)
    assert len(notes) == len(fake_notes['notes'])
    assert original_notes == fake_notes  # ensure the original dict is not modified
    for note, note_dict in zip(notes, fake_notes['notes'], strict=True):
        # assert actual == expected
        assert dict(note.subject_metadata) == fake_notes['subject_metadata']
        # the note_metadata has additional values (e.g. reference, priority) that are found on
        # individual notes in the dict/yaml
        note_metadata = deepcopy(dict(note.note_metadata))
        actual_reference = note_metadata.pop('reference', None)
        actual_priority = note_metadata.pop('priority', 'medium')
        # after removing reference/priority, the remaining values should match the original yaml
        assert note_metadata == fake_notes['note_metadata']
        if isinstance(note, TextNote):
            assert note.text() == dedent(note_dict['text']).strip()
        elif isinstance(note, Flashcard):
            if 'term' in note_dict:
                expected_preview = dedent(note_dict['term']).strip()
                expected_answer = dedent(note_dict['definition']).strip()
            elif 'question' in note_dict:
                expected_preview = dedent(note_dict['question']).strip()
                expected_answer = dedent(note_dict['answer']).strip()
            else:
                raise ValueError("Invalid note format.")
            assert note.preview() == expected_preview
            assert note.answer() == expected_answer
            assert note.text() == expected_preview + expected_answer
    
        assert note.note_metadata.reference == actual_reference
        assert note.note_metadata.reference == note_dict.get('reference', None)
        assert note.note_metadata.priority == Priority[actual_priority]
        assert Priority[actual_priority] == Priority[note_dict.get('priority', 'medium')]
        assert note.note_metadata.tags == fake_notes['note_metadata']['tags']
