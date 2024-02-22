import faiss
import numpy as np
import torch
from sentence_transformers import CrossEncoder


from retrieval.Ranker import Ranker


class CrossEncodingRanker(Ranker):
    def __init__(self, top_k: int, name=None):
        """
        :param top_k: The number of chunks to return
        :type top_k: int

        :param kwargs: Keyword arguments for the TfidfVectorizer
        :type kwargs: dict
        """
        super().__init__(top_k, name)
        self.index = None
        if name is None:
            self.name += "_sent_embedding"

        device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', device=device)

    def init_chunks(self, chunks: list[str]):
        self.chunks: list[str] = chunks

    def rank(self, query: str, return_distances: bool = False) -> list[str] | list[tuple[str, float]]:
        scores = self.model.predict([[query, chunk] for chunk in self.chunks])
        indices = np.argsort(scores)[::-1]
        if return_distances:
            return [(self.chunks[i], scores[i]) for i in indices]
        return [self.chunks[i] for i in indices[:self.top_k]]

    def batch_rank(self, queries: list[str], batch_size: int = 100) -> list[list[str]]:
        scores = self.model.predict([[query, chunk] for chunk in self.chunks for query in queries], batch_size=batch_size)
        scores = scores.reshape(len(queries), len(self.chunks))
        indices = np.argsort(scores, axis=1)[:, ::-1]
        return [[self.chunks[i] for i in indices[j][:self.top_k]] for j in range(len(queries))]
