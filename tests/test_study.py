"""Tests for CLI."""

import shutil
import os
from click.testing import CliRunner
from source.library.notes import Flashcard, Note
from study import cli, filter_notes, load_history, load_notes


def test__load_notes():  # noqa
    notes = load_notes("/code/tests/test_files/fake_notes.yaml")
    assert isinstance(notes, list)
    assert len(notes) > 1
    assert isinstance(notes[0], Note)


def test__filter_notes__flashcards():  # noqa
    original_notes = load_notes("/code/tests/test_files/fake_notes_.yaml")
    notes = filter_notes(original_notes, flash_only=True)
    assert all(isinstance(note, Flashcard) for note in notes)


def test__filter_notes__category():  # noqa
    original_notes = load_notes("/code/tests/test_files/fake_notes*.yaml")
    notes = filter_notes(original_notes, category="OMSCS - filtered")
    assert len(notes) == len(original_notes) / 2
    assert all(note.subject_metadata.category == "OMSCS - filtered" for note in notes)


def test__filter_notes__ident():  # noqa
    original_notes = load_notes("/code/tests/test_files/fake_notes*.yaml")
    notes = filter_notes(original_notes, ident="CS 6200 - filtered")
    assert len(notes) == len(original_notes) / 2
    assert notes[0].subject_metadata.ident == "CS 6200 - filtered"


def test__filter_notes__name():  # noqa
    original_notes = load_notes("/code/tests/test_files/fake_notes*.yaml")
    notes = filter_notes(original_notes, name="Graduate Introduction to Operating Systems - filtered")  # noqa
    assert len(notes) == len(original_notes) / 2
    assert notes[0].subject_metadata.name == "Graduate Introduction to Operating Systems - filtered"  # noqa


def test__filter_notes__abbr():  # noqa
    original_notes = load_notes("/code/tests/test_files/fake_notes*.yaml")
    notes = filter_notes(original_notes, abbr="GIOS - filtered")
    assert len(notes) == len(original_notes) / 2
    assert notes[0].subject_metadata.abbreviation == "GIOS - filtered"


def test__cycle__defaults__no_history():  # noqa
    runner = CliRunner()
    result = runner.invoke(
        cli,
        ['cycle', '--notes_path', '/code/tests/test_files/fake_notes.yaml'],
        # 'q' will work regardless of note type that is drawn
        input='q\n',
    )
    assert result.exit_code == 0


def test__cycle__flash_only__no_history():  # noqa
    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            'cycle',
            '--flash_only',
            '--notes_path', '/code/tests/test_files/fake_notes.yaml',
        ],
        # this sequence will only work with Flashcard notes
        input='1\nq\n',
    )
    assert result.exit_code == 0

def test__cycle__flash_only__with_history():  # noqa
    notes_path = "/code/tests/test_files/fake_notes.yaml"
    fake_history_path = "/code/tests/test_files/fake_history_no_history.yaml"
    temp_history_path = "/code/tests/test_files/temp___fake_history.yaml"
    shutil.copy(fake_history_path, temp_history_path)

    orginal_notes = load_notes(notes_path)
    expected_uuids = {note.uuid() for note in orginal_notes}
    non_flashcard_uuids = {note.uuid() for note in orginal_notes if not isinstance(note, Flashcard)}  # noqa
    flashcard_uuids = expected_uuids - non_flashcard_uuids

    try:
        ####
        # test with 2 correct answers; 0 incorrect answers
        ####
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                'cycle',
                '--flash_only',
                '--notes_path', notes_path,
                '--history_path', temp_history_path,
            ],
            # this sequence will only work with Flashcard notes
            input='1\ny\n1\ny\nq\n',
        )
        assert result.exit_code == 0
        saved_history = load_history(temp_history_path)
        # ensure history has not changed for uuids that are not flashcards and that they still
        # exist and have not been removed.
        assert set(saved_history.keys()) == expected_uuids
        for uuid in non_flashcard_uuids:
            assert saved_history[uuid].correct == 0
            assert saved_history[uuid].incorrect == 0

        flashcard_history = [h for uuid, h in saved_history.items() if uuid in flashcard_uuids]
        assert sum(h.correct for h in flashcard_history) == 2
        assert sum(h.incorrect for h in flashcard_history) == 0

        ####
        # test with 0 correct answers; 2 incorrect answers
        ####
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                'cycle',
                '--flash_only',
                '--notes_path', notes_path,
                '--history_path', temp_history_path,
            ],
            # this sequence will only work with Flashcard notes
            input='1\nn\n1\nn\nq\n',
        )
        assert result.exit_code == 0
        saved_history = load_history(temp_history_path)
        # ensure history has not changed for uuids that are not flashcards and that they still
        # exist and have not been removed.
        assert set(saved_history.keys()) == expected_uuids
        for uuid in non_flashcard_uuids:
            assert saved_history[uuid].correct == 0
            assert saved_history[uuid].incorrect == 0

        flashcard_history = [h for uuid, h in saved_history.items() if uuid in flashcard_uuids]
        assert sum(h.correct for h in flashcard_history) == 2  # from previous test/history
        assert sum(h.incorrect for h in flashcard_history) == 2

        ####
        # test with 0 correct answers; 2 incorrect answers
        ####
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                'cycle',
                '--flash_only',
                '--notes_path', notes_path,
                '--history_path', temp_history_path,
            ],
            # this sequence will only work with Flashcard notes
            input='1\ny\n1\nn\n1\nn\n1\ny\nq\n',
        )
        assert result.exit_code == 0
        saved_history = load_history(temp_history_path)
        # ensure history has not changed for uuids that are not flashcards and that they still
        # exist and have not been removed.
        assert set(saved_history.keys()) == expected_uuids
        for uuid in non_flashcard_uuids:
            assert saved_history[uuid].correct == 0
            assert saved_history[uuid].incorrect == 0

        flashcard_history = [h for uuid, h in saved_history.items() if uuid in flashcard_uuids]
        # 2 correct and 2 incorrects answers from previous history file
        assert sum(h.correct for h in flashcard_history) == 4
        assert sum(h.incorrect for h in flashcard_history) == 4

    finally:
        os.remove(temp_history_path)
