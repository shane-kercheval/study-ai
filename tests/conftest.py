"""Defines test fixtures for pytest unit-tests."""
import pytest
import yaml


@pytest.fixture
def fake_notes() -> dict:
    with open("/code/tests/test_files/fake_notes.yaml") as h:
        return yaml.safe_load(h)
