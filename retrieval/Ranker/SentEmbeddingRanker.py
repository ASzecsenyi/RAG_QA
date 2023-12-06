import faiss
import numpy as np
import torch
from sentence_transformers import SentenceTransformer


from retrieval.Ranker import Ranker


class SentEmbeddingRanker(Ranker):
    def __init__(self, top_k: int, **kwargs):
        """
        :param top_k: The number of chunks to return
        :type top_k: int

        :param kwargs: Keyword arguments for the TfidfVectorizer
        :type kwargs: dict
        """
        super().__init__(top_k)
        self.index = None
        self.name += f"_{kwargs}"
        self.vectors = None

        device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model = SentenceTransformer('paraphrase-MiniLM-L6-v2', device=device)

    def init_chunks(self, chunks: list[str]):
        self.chunks: list[str] = chunks
        vectors = self.model.encode(self.chunks)

        embedding_size = len(vectors[0])

        chunks_arr: np.ndarray = np.vstack(vectors, dtype="float32")

        # #### FAISS ####
        self.index = faiss.index_factory(embedding_size, "Flat", faiss.METRIC_INNER_PRODUCT)
        self.index.add(chunks_arr)

    def rank(self, query: str) -> list[str]:
        query_vector = self.model.encode([query])
        query_arr: np.ndarray = np.vstack(query_vector, dtype="float32")
        _, indices = self.index.search(query_arr, self.top_k)
        return [self.chunks[i] for i in indices[0]]

    def batch_rank(self, queries: list[str], batch_size: int = 100) -> list[list[str]]:
        query_vectors = self.model.encode(queries)
        query_arr: np.ndarray = np.vstack(query_vectors, dtype="float32")
        _, indices = self.index.search(query_arr, self.top_k)
        return [[self.chunks[i] for i in indices[j]] for j in range(len(queries))]