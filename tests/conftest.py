"""Defines test fixtures for pytest unit-tests."""
import pytest
import yaml

from source.library.notes import History


@pytest.fixture()
def fake_notes() -> dict:
    """Return a dictionary of fake notes."""
    with open('tests/test_files/fake_notes.yaml') as h:
        return yaml.safe_load(h)


@pytest.fixture()
def invalid_notes_no_uuids() -> dict:
    """Return a dictionary of fake notes."""
    with open('tests/test_files/invalid_notes_no_uuids.yaml') as h:
        return yaml.safe_load(h)


@pytest.fixture()
def fake_history() -> dict:
    """Return a dictionary of fake history."""
    with open('tests/test_files/fake_history.yaml') as h:
        history = yaml.safe_load(h)
        return{
            uuid: History([True]*answers['correct'] + [False]*answers['incorrect'])
            for uuid, answers in history.items()
        }


@pytest.fixture()
def fake_history_equal() -> dict:
    """Return a dictionary of fake history."""
    with open('tests/test_files/fake_history_equal.yaml') as h:
        history = yaml.safe_load(h)
        return{
            uuid: History([True]*answers['correct'] + [False]*answers['incorrect'])
            for uuid, answers in history.items()
        }
