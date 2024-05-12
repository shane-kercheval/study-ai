"""Test utilities.py."""

from source.library.notes import parse


def test__parse():
    import yaml
    with open("/code/tests/test_files/notes1.yaml") as _handle:
        yaml_data = yaml.safe_load(_handle)
    notes = parse(yaml_data)
    for note in notes:
        assert note.subject_metadata == yaml_data['subject_metadata']
        assert note.note_metadata == yaml_data['note_metadata']
        assert note.note
    notes
    