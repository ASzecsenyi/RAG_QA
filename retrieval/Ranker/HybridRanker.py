import faiss
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
import nltk
import contractions
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer

from retrieval.Ranker import Ranker
from retrieval.Ranker.SentEmbeddingRanker import SentEmbeddingRanker
from retrieval.Ranker.TfidfRanker import TfidfRanker

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
# if not nltk.data.find('corpora/wordnet'):
#    nltk.download('wordnet')


class HybridRanker(Ranker):

    def __init__(self, top_k: int, name=None,
                 sparse: Ranker = TfidfRanker(top_k=5),
                 dense: Ranker = SentEmbeddingRanker(top_k=5),
                 sparse_weight: float = 0.5,
                 **kwargs):
        """
        :param top_k: The number of chunks to return
        :type top_k: int

        :param kwargs: Keyword arguments for the TfidfVectorizer
        :type kwargs: dict
        """
        super().__init__(top_k, name)
        if name is None:
            self.name += "_hybrid"

        self.sparse_ranker = sparse
        self.dense_ranker = dense
        self.sparse_weight = sparse_weight

    def init_chunks(self, chunks: list[str]):
        self.chunks = chunks
        self.sparse_ranker.init_chunks(chunks)
        self.dense_ranker.init_chunks(chunks)

    def rank(self, query: str, return_distances: bool = False) -> list[str] | list[tuple[str, float]]:
        # [(chunk, sim), ...]
        sparse_rank: list[tuple[str, float]] = self.sparse_ranker.rank(query, return_distances=True)
        dense_rank: list[tuple[str, float]] = self.dense_ranker.rank(query, return_distances=True)

        # print("")
        # print(sparse_rank[:5])
        # print(dense_rank[:5])
        # print("")

        # make sure all chunks appear in both rankings
        sparse_chunks = [x[0] for x in sparse_rank]
        dense_chunks = [x[0] for x in dense_rank]
        assert set(sparse_chunks) == set(dense_chunks) == set(self.chunks), f"Chunks in sparse and dense rankings do not match. {set(sparse_chunks) - set(dense_chunks)} in sparse but not dense, {set(dense_chunks) - set(sparse_chunks)} in dense but not sparse."

        # normalize rankings
        sparse_rank_max = max(sparse_rank, key=lambda x: x[1])[1]
        sparse_rank_min = min(sparse_rank, key=lambda x: x[1])[1]
        dense_rank_max = max(dense_rank, key=lambda x: x[1])[1]
        dense_rank_min = min(dense_rank, key=lambda x: x[1])[1]

        sparse_rank = [(x[0], (x[1] - sparse_rank_min) / (sparse_rank_max - sparse_rank_min)) for x in sparse_rank]
        dense_rank = [(x[0], (x[1] - dense_rank_min) / (dense_rank_max - dense_rank_min)) for x in dense_rank]

        # print("")
        # print(sparse_rank[:5])
        # print(dense_rank[:5])
        # print("")

        # Combine the two rankings
        combined_rank = []
        for i in range(len(sparse_rank)):
            current_chunk = sparse_rank[i][0]
            sparse_sim = sparse_rank[i][1]
            dense_sim = dense_rank[dense_chunks.index(current_chunk)][1]
            combined_sim = self.sparse_weight * sparse_sim + (1 - self.sparse_weight) * (1 - dense_sim)
            combined_rank.append((current_chunk, combined_sim))
        combined_rank.sort(key=lambda x: x[1], reverse=True)

        if return_distances:
            return combined_rank

        return [x[0] for x in combined_rank[:self.top_k]]

    def batch_rank(self, queries: list[str], batch_size: int = 100) -> list[list[str]]:
        raise NotImplementedError
