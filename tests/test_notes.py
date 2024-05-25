"""Test notes.py."""

from copy import deepcopy
from textwrap import dedent
import numpy as np
from source.library.notes import (
    DefinitionNote,
    Flashcard,
    History,
    NoteMetadata,
    Priority,
    QuestionAnswerNote,
    SubjectMetadata,
    TextNote,
    TestBank,
    parse,
)


def test__Note__uuid__default_values():  # noqa
    # test with subset of metadata fields
    note = TextNote(
        text="test text",
        subject_metadata=SubjectMetadata(name='test category'),
        note_metadata=NoteMetadata(source_name='test source'),
    )
    # uuid should be deterministic across machines and runs
    assert note.uuid() == 'bfa717e88b323890fc19c2d8a29b44d145b8c841a5fdf22ceb5bc1aba1fc5d1f'


def test__Note__uuid__additional_values():  # noqa
    # test with subset of metadata fields
    note = TextNote(
        text="test text",
        subject_metadata=SubjectMetadata(name='test category', ident='test ident'),
        note_metadata=NoteMetadata(source_name='test source', source_reference='test reference', tags=['tag1', 'tag2']),  # noqa
    )
    # uuid should be deterministic across machines and runs
    assert note.uuid() == '80484b1c2633e9ab82e599937a43ef77e177d498f7749b4b468a5b4d74cf70c3'


def test__parse(fake_notes):   # noqa
    original_notes = deepcopy(fake_notes)
    notes = parse(fake_notes)
    assert len(notes) == len(fake_notes['notes'])
    assert original_notes == fake_notes  # ensure the original dict is not modified

    expected_test_types = {TextNote, DefinitionNote, QuestionAnswerNote}
    found_test_types = set()
    for note, note_dict in zip(notes, fake_notes['notes'], strict=True):
        found_test_types.add(type(note))
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
    assert expected_test_types == found_test_types


def test__history__answer():  # noqa
    history = History()
    assert history.correct == 0
    assert history.incorrect == 0
    assert history.to_dict() == {'correct': 0, 'incorrect': 0}
    history.answer(correct=True)
    assert history.correct == 1
    assert history.incorrect == 0
    assert history.to_dict() == {'correct': 1, 'incorrect': 0}
    history.answer(correct=False)
    assert history.correct == 1
    assert history.incorrect == 1
    assert history.to_dict() == {'correct': 1, 'incorrect': 1}
    history.answer(correct=True)
    assert history.correct == 2
    assert history.incorrect == 1
    assert history.to_dict() == {'correct': 2, 'incorrect': 1}
    history.answer(correct=False)
    assert history.correct == 2
    assert history.incorrect == 2
    assert history.to_dict() == {'correct': 2, 'incorrect': 2}

    history_copy = History(**history.to_dict())
    assert history.correct == history_copy.correct
    assert history.incorrect == history_copy.incorrect
    assert history.to_dict() == history_copy.to_dict()


def test__history__beta_draw_no_history():  # noqa
    # The average probability of a draw associated with no history, over many draws should be close
    # to 0.5 (50/50 chance of being correct/incorrect)
    history = History()
    # test that beta_draw works on 0 correct & 0 incorrect
    draws = [history.beta_draw() for _ in range(10000)]
    assert all(0 <= draw <= 1 for draw in draws)
    # should be a wide probability distribution for a new note with no history
    assert any(draw < 0.1 for draw in draws)
    assert any(draw > 0.9 for draw in draws)
    assert np.isclose(np.mean(draws), 0.5, atol=0.05)


def test__history__beta_draw_confident_history():  # noqa
    # The probability of a draw associated with an instance with a lot of history should be very
    # close to the actual probability of being correct
    history = History(correct=100000, incorrect=100000)
    draws = [history.beta_draw() for _ in range(10000)]
    assert all(np.isclose(draw, 0.5, atol=0.01) for draw in draws)

    history = History(correct=100000, incorrect=0)
    draws = [history.beta_draw() for _ in range(10000)]
    assert all(np.isclose(draw, 1, atol=0.01) for draw in draws)

    history = History(correct=0, incorrect=100000)
    draws = [history.beta_draw() for _ in range(10000)]
    assert all(np.isclose(draw, 0, atol=0.01) for draw in draws)


def test__history__beta_draw_no_history__seed():  # noqa
    # The average probability of a draw associated with no history, over many draws should be close
    # to 0.5 (50/50 chance of being correct/incorrect)
    history = History()
    # test that beta_draw works on 0 correct & 0 incorrect
    draw = history.beta_draw(seed=42)
    assert 0 <= draw <= 1
    assert draw == history.beta_draw(seed=42)
    assert draw == history.beta_draw(seed=42)


def test__TestBank__no_history(fake_notes):  # noqa
    test_bank = TestBank(notes=parse(fake_notes))
    assert len(test_bank) == len(fake_notes['notes'])
    draws = [test_bank.draw().uuid() for _ in range(len(test_bank) * 1000)]
    expected_uuids = {note['uuid'] for note in test_bank.notes.values()}
    assert set(draws) == expected_uuids
    # Each note should be drawn roughly the same number of times
    # get counts of each uuid
    counts = {uuid: draws.count(uuid) for uuid in expected_uuids}
    # check that the counts are roughly equal
    expected_count = len(draws) / len(expected_uuids)
    assert all(
        0.9 * expected_count <= count <= 1.1 * expected_count
        for count in counts.values()
    )
    # test history() method returns correct uuids and counts
    history = test_bank.history()
    assert history.keys() == expected_uuids
    assert all(history[uuid].correct == 0 for uuid in expected_uuids)
    assert all(history[uuid].incorrect == 0 for uuid in expected_uuids)
