"""Defines test fixtures for pytest unit-tests."""
import pytest
import yaml

from study import load_history


@pytest.fixture()
def fake_notes() -> dict:
    """Return a dictionary of fake notes."""
    with open("tests/test_files/fake_notes.yaml") as h:
        return yaml.safe_load(h)


@pytest.fixture()
def fake_history() -> dict:
    """Return a dictionary of fake history."""
    return load_history("tests/test_files/fake_history.yaml")


@pytest.fixture()
def fake_history_equal() -> dict:
    """Return a dictionary of fake history."""
    return load_history("tests/test_files/fake_history_equal.yaml")
