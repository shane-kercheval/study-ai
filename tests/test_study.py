"""Tests for CLI."""

import shutil
import os
from click.testing import CliRunner
import pytest
from source.library.notes import Flashcard, Note
from source.cli.utilities import filter_notes, load_history, load_notes
from study import cli


def test__load_notes():  # noqa
    notes = load_notes('tests/test_files/fake_notes.yaml', generate_save_uuids=False)
    assert isinstance(notes, list)
    assert len(notes) > 1
    assert isinstance(notes[0], Note)


def test__load_notes__invalid_directory_file__should_raise_exception():  # noqa
    with pytest.raises(FileNotFoundError):
        load_notes('tests/test_files/file_does_not_exist.yaml', generate_save_uuids=False)


def test__load_notes__directory_does_not_exist__should_raise_exception():  # noqa
    with pytest.raises(FileNotFoundError):
        load_notes('tests/directory_does_not_exist/*.yaml', generate_save_uuids=False)


def test__load_notes__no_files__should_raise_exception():  # noqa
    with pytest.raises(FileNotFoundError):
        load_notes('tests/test_files/*.no_extensions', generate_save_uuids=False)


def test__filter_notes__flashcards():  # noqa
    original_notes = load_notes('tests/test_files/fake_notes*.yaml', generate_save_uuids=False)
    notes = filter_notes(original_notes, flash_only=True)
    assert all(isinstance(note, Flashcard) for note in notes)


def test__filter_notes__category():  # noqa
    original_notes = load_notes('tests/test_files/fake_notes*.yaml', generate_save_uuids=False)
    notes = filter_notes(original_notes, category='OMSCS - filtered')
    assert len(notes) == len(original_notes) / 2
    assert all(note.subject_metadata.category == 'OMSCS - filtered' for note in notes)


def test__filter_notes__ident():  # noqa
    original_notes = load_notes('tests/test_files/fake_notes*.yaml', generate_save_uuids=False)
    notes = filter_notes(original_notes, ident='CS 6200 - filtered')
    assert len(notes) == len(original_notes) / 2
    assert notes[0].subject_metadata.ident == 'CS 6200 - filtered'


def test__filter_notes__name():  # noqa
    original_notes = load_notes('tests/test_files/fake_notes*.yaml', generate_save_uuids=False)
    notes = filter_notes(original_notes, name='Graduate Introduction to Operating Systems - filtered')  # noqa
    assert len(notes) == len(original_notes) / 2
    assert notes[0].subject_metadata.name == 'Graduate Introduction to Operating Systems - filtered'  # noqa


def test__filter_notes__abbr():  # noqa
    original_notes = load_notes('tests/test_files/fake_notes*.yaml', generate_save_uuids=False)
    notes = filter_notes(original_notes, abbr='GIOS - filtered')
    assert len(notes) == len(original_notes) / 2
    assert notes[0].subject_metadata.abbreviation == 'GIOS - filtered'


def test__study__defaults():  # noqa
    # tests loading in all notes saved in the default notes path
    runner = CliRunner()
    result = runner.invoke(
        cli,
        ['study'],
        # 'q' will work regardless of note type that is drawn
        input='q\n',
    )
    assert result.exit_code == 0


def test__study__multiple_notes_paths():  # noqa
    # tests loading multiple notes paths
    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            'study',
            '--notes_paths', 'tests/test_files/fake_notes.yaml',
            '--notes_paths', 'tests/test_files/fake_notes_filtering.yaml',
        ],
        # 'q' will work regardless of note type that is drawn
        input='q\n',
    )
    assert result.exit_code == 0
    # ensure all notes are loaded
    assert "Available notes: 8" in result.output


def test__study__defaults__no_history():  # noqa
    runner = CliRunner()
    result = runner.invoke(
        cli,
        ['study', '--notes_paths', 'tests/test_files/fake_notes.yaml'],
        # 'q' will work regardless of note type that is drawn
        input='q\n',
    )
    assert result.exit_code == 0


def test__study__flash_only__no_history():  # noqa
    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            'study',
            '--flash_only',
            '--notes_paths', 'tests/test_files/fake_notes.yaml',
        ],
        # this sequence will only work with Flashcard notes
        input='1\nq\n',
    )
    assert result.exit_code == 0


def test__study__flash_only__with_history():  # noqa
    notes_paths = 'tests/test_files/fake_notes.yaml'
    fake_history_path = 'tests/test_files/fake_history_no_history.yaml'
    temp_history_path = 'tests/test_files/temp___fake_history.yaml'
    shutil.copy(fake_history_path, temp_history_path)

    orginal_notes = load_notes(notes_paths)
    expected_uuids = {note.uuid for note in orginal_notes}
    non_flashcard_uuids = {note.uuid for note in orginal_notes if not isinstance(note, Flashcard)}
    flashcard_uuids = expected_uuids - non_flashcard_uuids

    try:
        ####
        # test with 2 correct answers; 0 incorrect answers
        ####
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                'study',
                '--flash_only',
                '--notes_paths', notes_paths,
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
            assert saved_history[uuid].answers == []

        flashcard_history = [h for uuid, h in saved_history.items() if uuid in flashcard_uuids]
        assert sum(sum(h.answers) for h in flashcard_history) == 2  # 2 correct answers
        assert sum(len(h.answers) for h in flashcard_history) == 2  # 2 total answers

        ####
        # test with 0 correct answers; 2 incorrect answers
        # this will test that the history is updated correctly across multiple runs
        ####
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                'study',
                '--flash_only',
                '--notes_paths', notes_paths,
                '--history_path', temp_history_path,
            ],
            # this sequence will only work with Flashcard notes
            input='1\nn\nq\n',
        )
        assert result.exit_code == 0
        saved_history = load_history(temp_history_path)
        # ensure history has not changed for uuids that are not flashcards and that they still
        # exist and have not been removed.
        assert set(saved_history.keys()) == expected_uuids
        for uuid in non_flashcard_uuids:
            assert saved_history[uuid].answers == []

        flashcard_history = [h for uuid, h in saved_history.items() if uuid in flashcard_uuids]
        assert sum(sum(h.answers) for h in flashcard_history) == 2  # from previous test/history
        assert sum(len(h.answers) for h in flashcard_history) == 3  # 3 total answers

        ####
        # test with 0 correct answers; 2 incorrect answers
        # this will test that the history is updated correctly across multiple runs
        ####
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                'study',
                '--flash_only',
                '--notes_paths', notes_paths,
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
            assert saved_history[uuid].answers == []

        flashcard_history = [h for uuid, h in saved_history.items() if uuid in flashcard_uuids]
        # 2 correct and 2 incorrects answers from previous history file
        assert sum(sum(h.answers) for h in flashcard_history) == 4
        assert sum(len(h.answers) for h in flashcard_history) == 7  # 7 total answers

    finally:
        os.remove(temp_history_path)


def test__search():  # noqa
    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            'search',
        ],
        input='What are ip addresses?\nq\n',
    )
    assert result.exit_code == 0
    assert 'Cosine Similarity' in result.output
    # ensure this note is returned in the search results
    assert '1e16ee44-77e0-47d7-af51-e82980a6ff64' in result.output
