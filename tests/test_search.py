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


def test__VectorDatabase__(fake_notes) -> None:  # noqa
    notes = dict_to_notes(fake_notes)
    db_path = 'temp___test_vector_database.parquet'
    try:
        db = VectorDatabase(db_path=db_path)
        db.add(notes=notes[:-1], save=False)
        assert db.df['uuid'].tolist() == [note.uuid() for note in notes[:-1]]
        assert db.df['text'].tolist() == [note.text() for note in notes[:-1]]
        assert db.df.apply(lambda x: np.array(x['embedding']).shape[0] > 1, axis=1).all()

        # for the first note, we are going to update the text, which should trigger an update of the
        # embedding if we pass in the notes again; the 2nd and 3rd embeddings should not change;
        # the last note should be added to the database
        embedding_note_1 = db.df.loc[db.df['uuid'] == notes[0].uuid(), 'embedding'].to_numpy()[0]
        embedding_note_2 = db.df.loc[db.df['uuid'] == notes[1].uuid(), 'embedding'].to_numpy()[0]
        embedding_note_3 = db.df.loc[db.df['uuid'] == notes[2].uuid(), 'embedding'].to_numpy()[0]

        notes[0]._term = 'new term'
        db.add(notes=notes, save=True)  # update 1st note, add 4th note; save the database

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



