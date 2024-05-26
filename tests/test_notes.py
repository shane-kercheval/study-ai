"""Test notes.py."""

from copy import deepcopy
from textwrap import dedent
import numpy as np
import pytest
from source.library.utilities import softmax_dict
from source.library.notes import (
    DefinitionNote,
    Flashcard,
    History,
    Priority,
    QuestionAnswerNote,
    TextNote,
    NoteBank,
    add_uuids_to_dict,
    dict_to_notes,
)
from study import load_notes


def test__dict_to_notes(fake_notes):   # noqa
    original_notes = deepcopy(fake_notes)
    notes = dict_to_notes(fake_notes)
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
        actual_priority = note_dict.pop('priority', 'medium')
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
            assert note.text() == expected_preview + " " + expected_answer
        assert note.note_metadata.reference == actual_reference
        assert note.note_metadata.reference == note_dict.get('reference', None)
        assert isinstance(note.priority, Priority)
        assert note.priority == Priority[actual_priority]
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


def test__history__success_probability_no_history():  # noqa
    # The average probability of a draw associated with no history, over many draws should be close
    # to 0.5 (50/50 chance of being correct/incorrect)
    history = History()
    # test that success_probability works on 0 correct & 0 incorrect
    draws = [history.probability_correct() for _ in range(10000)]
    assert all(0 <= draw <= 1 for draw in draws)
    # should be a wide probability distribution for a new note with no history
    assert any(draw < 0.1 for draw in draws)
    assert any(draw > 0.9 for draw in draws)
    assert np.isclose(np.mean(draws), 0.5, atol=0.05)


def test__history__success_probability_confident_history():  # noqa
    # The probability of a draw associated with an instance with a lot of history should be very
    # close to the actual probability of being correct
    history = History(correct=100000, incorrect=100000)
    draws = [history.probability_correct() for _ in range(10000)]
    assert all(np.isclose(draw, 0.5, atol=0.01) for draw in draws)

    history = History(correct=100000, incorrect=0)
    draws = [history.probability_correct() for _ in range(10000)]
    assert all(np.isclose(draw, 1, atol=0.01) for draw in draws)

    history = History(correct=0, incorrect=100000)
    draws = [history.probability_correct() for _ in range(10000)]
    assert all(np.isclose(draw, 0, atol=0.01) for draw in draws)


def test__history__success_probability_no_history__seed():  # noqa
    # The average probability of a draw associated with no history, over many draws should be close
    # to 0.5 (50/50 chance of being correct/incorrect)
    history = History()
    # test that success_probability works on 0 correct & 0 incorrect
    draw = history.probability_correct(seed=42)
    assert 0 <= draw <= 1
    assert draw == history.probability_correct(seed=42)
    assert draw == history.probability_correct(seed=42)


@pytest.mark.parametrize("history", [None, {}])
def test__TestBank__no_history__expect_equal_draws(fake_notes, history):  # noqa
    """
    Test that the TestBank draws notes with roughly equal probability when no history is
    present.
    """
    test_bank = NoteBank(notes=dict_to_notes(fake_notes), history=history)
    {uuid: n['note'].text() for uuid, n in test_bank.notes.items()}
    assert len(test_bank) == len(fake_notes['notes'])
    draws = [test_bank.draw().uuid for _ in range(len(test_bank) * 1000)]
    expected_uuids = {note['note'].uuid for note in test_bank.notes.values()}
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
    assert {uuid: h.to_dict() for uuid, h in test_bank.history().items()} == test_bank.history(to_dict=True)  # noqa
    history = test_bank.history()
    assert history.keys() == expected_uuids
    assert all(history[uuid].correct == 0 for uuid in expected_uuids)
    assert all(history[uuid].incorrect == 0 for uuid in expected_uuids)


@pytest.mark.parametrize("history", [None, {}])
def test__TestBank__no_history__answer__history_updates_correctly(fake_notes, history):  # noqa
    """Test that the history is updated correctly when answering questions."""
    test_bank = NoteBank(notes=dict_to_notes(fake_notes), history=history)
    assert len(test_bank) == len(fake_notes['notes'])
    expected_uuids = [note['note'].uuid for note in test_bank.notes.values()]
    assert all(test_bank.history()[uuid].correct == 0 for uuid in expected_uuids)
    assert all(test_bank.history()[uuid].incorrect == 0 for uuid in expected_uuids)
    for index in range(len(test_bank)):
        test_bank.answer(expected_uuids[index], correct=True)
        assert test_bank.history()[expected_uuids[index]].correct == 1
        assert test_bank.history()[expected_uuids[index]].incorrect == 0
        test_bank.answer(expected_uuids[index], correct=False)
        assert test_bank.history()[expected_uuids[index]].correct == 1
        assert test_bank.history()[expected_uuids[index]].incorrect == 1
        test_bank.answer(expected_uuids[index], correct=True)
        assert test_bank.history()[expected_uuids[index]].correct == 2
        assert test_bank.history()[expected_uuids[index]].incorrect == 1
        assert {uuid: h.to_dict() for uuid, h in test_bank.history().items()} == test_bank.history(to_dict=True)  # noqa


def test__TestBank__draw__priority_weights(fake_notes, fake_history_equal):  # noqa
    """
    Test that the TestBank draws notes with probability corresponding to the priority weights. We
    are using a large number of draws to ensure that the counts are close to the expected. All
    histories contain the same number of correct and incorrect answers (i.e. probability of 50/50).
    Therefore the priority weights should be the only factor that determines the probability of
    drawing a note.
    """
    test_bank = NoteBank(notes=dict_to_notes(fake_notes), history=fake_history_equal)
    weights = {Priority.high: 4, Priority.medium: 2, Priority.low: 1}
    draws = [test_bank.draw(priority_weights=weights).uuid for _ in range(10000)]
    counts = {uuid: draws.count(uuid) for uuid in test_bank.notes}

    # we can do this since all histories have the same number of correct and incorrect answers
    uuid_weights = {uuid: weights[test_bank.notes[uuid]['note'].priority] for uuid in test_bank.notes}  # noqa
    expected_freq_lookups = softmax_dict(uuid_weights)
    # test that the counts are roughly equal to the expected probability of draw
    for uuid in test_bank.notes:
        # priority = test_bank.notes[uuid]['note'].priority
        expected_freq = expected_freq_lookups[uuid]
        assert np.isclose(counts[uuid] / len(draws), expected_freq, atol=0.02)


def test__TestBank__with_history__expect_draw_counts_to_correspond_with_history(fake_notes, fake_history):  # noqa
    """
    Test that the TestBank draws notes with probability corresponding to the history of the
    notes.
    """
    notes = dict_to_notes(fake_notes)
    expected_uuids = {n.uuid for n in notes}
    assert set(fake_history.keys()) == expected_uuids
    for history in fake_history.values():
        assert history.correct + history.incorrect == 100000

    test_bank = NoteBank(notes=notes, history=fake_history)
    assert len(test_bank) == len(notes)
    assert set(test_bank.notes.keys()) == expected_uuids

    # each note is associated with a beta distribution that has been updated 100,000 times
    # so the probability from success_probability should be very close to the actual probability of
    # getting the answer correct (which is inverse to the relative frequency it should be drawn)
    # we subtract from 1 because the beta distribution is the probability of getting the answer
    # correct, but we want the probability of drawing the note
    def prob_incorrect(history: History) -> float:
        correct = history.correct + 1
        incorrect = history.incorrect + 1
        return 1 - (correct / (correct + incorrect))

    expected_prob_of_draw = {uuid: prob_incorrect(history) for uuid, history in fake_history.items()}  # noqa
    expected_prob_of_draw = {
        uuid: prob / sum(expected_prob_of_draw.values())
        for uuid, prob in expected_prob_of_draw.items()
    }
    assert np.isclose(sum(expected_prob_of_draw.values()), 1, atol=0.0001)
    draws = [test_bank.draw().uuid for _ in range(len(test_bank) * 5000)]
    counts = {uuid: draws.count(uuid) for uuid in expected_uuids}
    # test that the counts are roughly equal to the expected probability of draw
    # we are using large number of draws to ensure that the counts are close to the expected
    for uuid in expected_uuids:
        assert np.isclose(counts[uuid] / len(draws), expected_prob_of_draw[uuid], atol=0.01)


def test__TestBank__duplicates_should_raise_exception():  # noqa
    original_notes = load_notes('tests/test_files/*fake_notes.yaml', generate_save_uuids=False)
    assert len({n.uuid for n in original_notes}) < len(original_notes)
    with pytest.raises(AssertionError):
        NoteBank(notes=original_notes, history=None)


def test__add_uuids_to_dict(invalid_notes_no_uuids):  # noqa
    invalid_copy = deepcopy(invalid_notes_no_uuids)
    with pytest.raises(TypeError, match="missing 1 required positional argument: 'uuid'"):
        dict_to_notes(invalid_notes_no_uuids)
    assert invalid_copy == invalid_notes_no_uuids
    valid_notes = add_uuids_to_dict(invalid_notes_no_uuids)
    assert all('uuid' in note for note in valid_notes['notes'])
    # this should not raise an error since the uuids have been added
    notes = dict_to_notes(valid_notes)
    for note_a, note_b in zip(notes, valid_notes['notes']):
        assert note_a.uuid == note_b['uuid']
