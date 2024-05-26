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

    Note that this test would break if Note removed cacheing from the uuid method because updating
    the text of the note would not trigger an update of the uuid.
    Related, we don't want our VectorDatabase to have implicit knowledge of the Note class, so we
    assume also test that the text for a particular uuid hasn't changed, since the uuid
    implementation could change in the future to allow uuids to remain static even if the text
    changes.
    """
    notes = dict_to_notes(fake_notes)
    db_path = 'temp___test_vector_database.parquet'
    try:
        db = VectorDatabase(db_path=db_path)
        changes = db.add(notes=notes[:-1], save=False)  # add all but the last note
        assert changes == {note.uuid(): 'added' for note in notes[:-1]}
        assert db.df['uuid'].tolist() == [note.uuid() for note in notes[:-1]]
        assert db.df['text'].tolist() == [note.text() for note in notes[:-1]]
        assert db.df.apply(lambda x: np.array(x['embedding']).shape[0] > 1, axis=1).all()

        # for the first note, we are going to update the text, which should trigger an update of
        # the embedding if we pass in the notes again; the 2nd and 3rd embeddings should not
        # change; the last note should be added to the database
        embedding_note_1 = db.df.loc[db.df['uuid'] == notes[0].uuid(), 'embedding'].to_numpy()[0]
        embedding_note_2 = db.df.loc[db.df['uuid'] == notes[1].uuid(), 'embedding'].to_numpy()[0]
        embedding_note_3 = db.df.loc[db.df['uuid'] == notes[2].uuid(), 'embedding'].to_numpy()[0]

        notes[0]._term = 'new term'  # doesn't trigger uuid update; uuid is cached in Note object
        changes = db.add(notes=notes, save=True)  # update 1st note, add 4th note; save database
        assert changes[notes[0].uuid()] == 'updated'
        assert changes[notes[3].uuid()] == 'added'

        # check that the database was saved correctly
        db_check = VectorDatabase(db_path=db_path)
        assert db_check.df['uuid'].tolist() == db.df['uuid'].tolist()
        assert db_check.df['text'].tolist() == db.df['text'].tolist()
        # check all embeddings match
        for i in range(len(db_check.df)):
            assert np.allclose(db_check.df['embedding'].iloc[i], db.df['embedding'].iloc[i])

        # check that the database was updated correctly
        # all notes should be in the database
        assert db.df['uuid'].tolist() == [note.uuid() for note in notes]
        assert db.df['text'].tolist() == [note.text() for note in notes]
        assert db.df.apply(lambda x: len(np.array(x['embedding']).shape) == 1, axis=1).all()
        assert db.df.apply(lambda x: np.array(x['embedding']).shape[0] > 1, axis=1).all()
        # the first note's embedding should have changed; the 2nd and 3rd should not have changed
        assert not np.allclose(embedding_note_1, db.df.loc[db.df['uuid'] == notes[0].uuid(), 'embedding'].tolist()[0])  # noqa
        assert np.allclose(embedding_note_2, db.df.loc[db.df['uuid'] == notes[1].uuid(), 'embedding'].tolist()[0])  # noqa
        assert np.allclose(embedding_note_3, db.df.loc[db.df['uuid'] == notes[2].uuid(), 'embedding'].tolist()[0])  # noqa
    finally:
        if os.path.exists(db_path):
            os.remove(db_path)


# def test__VectorDabase__search(fake_notes):
#     notes = dict_to_notes(fake_notes)
#     db_path = 'temp___test_vector_database.parquet'
#     try:
#         db = VectorDatabase(db_path=db_path)
#         db.add(notes=notes, save=False)

#     finally:
#         if os.path.exists(db_path):
#             os.remove(db_path)


