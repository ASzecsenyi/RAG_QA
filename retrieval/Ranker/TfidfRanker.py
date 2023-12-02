from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from retrieval.Ranker import Ranker


class TfidfRanker(Ranker):
    def __init__(self, top_k: int, **kwargs):
        """
        :param top_k: The number of chunks to return
        :type top_k: int

        :param kwargs: Keyword arguments for the TfidfVectorizer
        :type kwargs: dict
        """
        super().__init__(top_k)
        self.name += f"_{kwargs}"
        self.vectorizer = TfidfVectorizer(**kwargs)
        self.vectors = None

    def init_chunks(self, chunks: list[str]):
        self.chunks = chunks
        self.vectors = self.vectorizer.fit_transform(self.chunks)

    def rank(self, query: str) -> list[str]:
        query_vector = self.vectorizer.transform([query])
        cosine_similarities = cosine_similarity(query_vector, self.vectors).flatten()
        return [x for _, x in sorted(zip(cosine_similarities, self.chunks), reverse=True)][:self.top_k]
