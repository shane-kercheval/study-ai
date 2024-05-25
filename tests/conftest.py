"""Defines test fixtures for pytest unit-tests."""
import pytest
import yaml

from study import load_history


@pytest.fixture()
def fake_notes() -> dict:
    """Return a dictionary of fake notes."""
    with open("/code/tests/test_files/fake_notes.yaml") as h:
        return yaml.safe_load(h)

@pytest.fixture()
def fake_history() -> dict:
    """Return a dictionary of fake history."""
    return load_history("/code/tests/test_files/fake_history.yaml")
