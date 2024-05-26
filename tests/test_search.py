"""Test the search module."""
import os
import numpy as np
from source.library.notes import dict_to_notes
from source.library.search import VectorDatabase, cosine_similarity


def test__cosine_similarity() -> None:
    """Test the cosine similarity function."""
    x = [1, 2, 3]
    y = [-3, 0, 1]  # orthogonal to x
    # cosine similarity equals 0 when the vectors are orthogonal (90 degrees apart)
    # isclose is used here because of floating point precision
    assert np.isclose(cosine_similarity(x, y), 0.0, atol=1e-8)

    x = [1, 2, 3]
    y = [1, 2, 3]
    assert cosine_similarity(x, y) == 1.0


def test__VectorDatabase__creation_and_adding_notes(fake_notes) -> None:  # noqa
    """
    Test creating a VectorDatabase and adding notes to it. Tests adding new notes, and updating
    existing notes. Also tests loading the database from disk and checking that the notes were
    saved correctly.
    """
    notes = dict_to_notes(fake_notes)
    db_path = 'temp___test_vector_database.parquet'
    try:
        db = VectorDatabase(db_path=db_path)
        changes = db.add(notes=notes[:-1], save=False)  # add all but the last note
        assert changes == {note.uuid: 'added' for note in notes[:-1]}
        assert db._data['uuid'].tolist() == [note.uuid for note in notes[:-1]]
        assert db._data['text'].tolist() == [note.text() for note in notes[:-1]]
        assert db._data.apply(lambda x: np.array(x['embedding']).shape[0] > 1, axis=1).all()

        # for the first note, we are going to update the text, which should trigger an update of
        # the embedding if we pass in the notes again; the 2nd and 3rd embeddings should not
        # change; the last note should be added to the database
        embedding_note_1 = db._data.loc[db._data['uuid'] == notes[0].uuid, 'embedding'].to_numpy()[0]  # noqa
        embedding_note_2 = db._data.loc[db._data['uuid'] == notes[1].uuid, 'embedding'].to_numpy()[0]  # noqa
        embedding_note_3 = db._data.loc[db._data['uuid'] == notes[2].uuid, 'embedding'].to_numpy()[0]  # noqa

        notes[0]._term = 'new term'
        changes = db.add(notes=notes, save=True)  # update 1st note, add 4th note; save database
        assert changes[notes[0].uuid] == 'updated'
        assert changes[notes[3].uuid] == 'added'

        # check that the database was saved correctly
        db_check = VectorDatabase(db_path=db_path)
        assert db_check._data['uuid'].tolist() == db._data['uuid'].tolist()
        assert db_check._data['text'].tolist() == db._data['text'].tolist()
        # check all embeddings match
        for i in range(len(db_check._data)):
            assert np.allclose(db_check._data['embedding'].iloc[i], db._data['embedding'].iloc[i])

        # check that the database was updated correctly
        # all notes should be in the database
        assert db._data['uuid'].tolist() == [note.uuid for note in notes]
        assert db._data['text'].tolist() == [note.text() for note in notes]
        assert db._data.apply(lambda x: len(np.array(x['embedding']).shape) == 1, axis=1).all()
        assert db._data.apply(lambda x: np.array(x['embedding']).shape[0] > 1, axis=1).all()
        # the first note's embedding should have changed; the 2nd and 3rd should not have changed
        assert not np.allclose(embedding_note_1, db._data.loc[db._data['uuid'] == notes[0].uuid, 'embedding'].tolist()[0])  # noqa
        assert np.allclose(embedding_note_2, db._data.loc[db._data['uuid'] == notes[1].uuid, 'embedding'].tolist()[0])  # noqa
        assert np.allclose(embedding_note_3, db._data.loc[db._data['uuid'] == notes[2].uuid, 'embedding'].tolist()[0])  # noqa

        changes = db.add(notes=notes, save=True)  # no changes should be made
        assert changes == {}
    finally:
        if os.path.exists(db_path):
            os.remove(db_path)


def test__VectorDabase__search(fake_notes):  # noqa
    notes = dict_to_notes(fake_notes)
    db_path = 'temp___test_vector_database.parquet'
    try:
        db = VectorDatabase(db_path=db_path)
        db.add(notes=notes, save=False)

        # search for the most similar note to the first note
        original_df = db._data.copy()
        results = db.search(query=notes[0].text(), top_k=1)
        assert len(results) == 1
        assert results.loc[0, 'uuid'] == notes[0].uuid
        assert results.loc[0, 'cosine_similarity'] == 1.0
        # ensure that db.df was not modified
        assert db._data.equals(original_df)

        # search for the most similar note to the last note
        results = db.search(query=notes[-1].text(), top_k=2)
        assert len(results) == 2
        assert results['uuid'].tolist()[0] == notes[-1].uuid
        assert results['cosine_similarity'].tolist()[0] == 1.0
        # ensure that db.df was not modified
        assert db._data.equals(original_df)

        # search for the most similar note to a note that is not in the database
        # should still match against the last note, but the cosine similarity will be less <1.0
        results = db.search(query=notes[3].text() + " ???", top_k=100)
        assert len(results) == 4
        assert results['uuid'].tolist()[0] == notes[3].uuid
        assert 0.9 <= results['cosine_similarity'].tolist()[0] < 1.0
        # the second highest match should be the most similar note to the third note
        assert results['uuid'].tolist()[1] == notes[2].uuid
        # ensure that db.df was not modified
        assert db._data.equals(original_df)
    finally:
        if os.path.exists(db_path):
            os.remove(db_path)


