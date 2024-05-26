"""Classes and functions for creating/searching local vector database."""
# store vector database as parquet file (relatively small 'database' size, low overhead)
# create embeddings using SentenceTransformer
# use `all-MiniLM-L6-v2` model to generate embeddings; https://www.sbert.net/docs/pretrained_models.html
# relatively high quality, high speed, small model size
# it can be given a list of Note objects and will generate embeddings for each note, if the
# embedding doesn't already exist in the database
# the database will store uuid, text, and embedding for each note
# if the text is updated, the embedding will need to be updated as well


import os
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from source.library.notes import Note


def cosine_similarity(x: np.array, y: np.array) -> float:
    """Compute cosine similarity between two vectors."""
    return np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))


class VectorDatabase:
    """Class to store and search a vector database."""

    def __init__(self, db_path: str, model_name: str = 'all-MiniLM-L6-v2') -> None:
        """Initialize the vector database."""
        self._model = None
        self.model_name = model_name
        self.db_path = db_path
        if os.path.exists(db_path):
            self._data = pd.read_parquet(db_path)
            assert set(self._data.columns) == {'uuid', 'text', 'embedding'}
        else:
            self._data = pd.DataFrame(columns=['uuid', 'text', 'embedding'])
            self._data['embedding'] = self._data['embedding'].astype(object)  # allow for list of floats

    @property
    def model(self) -> SentenceTransformer:
        """Get the SentenceTransformer model."""
        if self._model is None:
            self._model = SentenceTransformer(self.model_name)
        return self._model


    def add(self, notes: list[Note], save: bool = True) -> dict[str, str]:
        """Add notes to the vector database."""
        new_notes = []
        changes = {}
        for note in notes:
            note_index = self._data['uuid'] == note.uuid
            assert note_index.sum() <= 1  # ensure there is at most one note with the same uuid
            note_exists = note_index.any()
            # if the note is already in the database and the text hasn't changed, skip
            if note_exists and note.text() == self._data.loc[note_index, 'text'].tolist()[0]:
                continue
            embedding = self.model.encode(note.text())
            if note_exists:
                # if the note is already in the database but the text has changed, update the text
                # and embedding; otherwise append the note to the new_notes list
                self._data.loc[note_index, 'text'] = note.text()
                # need to use `at` instead of `loc` to set the value with a sequence
                self._data.at[self._data.index[note_index][0], 'embedding'] = embedding  # noqa: PD008
                changes[note.uuid] = 'updated'
            else:
                new_notes.append(
                    {'uuid': note.uuid, 'text': note.text(), 'embedding': embedding},
                )
                changes[note.uuid] = 'added'
        self._data = pd.concat([self._data, pd.DataFrame(new_notes)], ignore_index=True)
        if save:
            self.save()
        return changes

    def search(self, query: str, top_k: int = 5) -> pd.DataFrame:
        """Search the vector database for the top k most similar vectors."""
        query_embedding = self.model.encode(query)
        self._data['cosine_similarity'] = self._data['embedding'].\
            apply(lambda x: cosine_similarity(query_embedding, x))
        results = self._data.sort_values('cosine_similarity', ascending=False).head(top_k).copy()
        self._data = self._data.drop(columns='cosine_similarity')
        return results

    def save(self) -> None:
        """Save the vector database to disk."""
        self._data.to_parquet(self.db_path, index=False)
