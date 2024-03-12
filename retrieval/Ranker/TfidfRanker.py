from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
import string
import nltk
import contractions
from nltk.stem import PorterStemmer

from retrieval.Ranker import Ranker


try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')


class TfidfRanker(Ranker):
    stop_words = set(stopwords.words('english'))
    stemmer = PorterStemmer()

    def __init__(self, top_k: int, name=None, **kwargs):
        """
        :param top_k: The number of chunks to return
        :type top_k: int

        :param kwargs: Keyword arguments for the TfidfVectorizer
        :type kwargs: dict
        """
        super().__init__(top_k, name)
        if name is None and not kwargs == {}:
            self.name += f"_{kwargs}"
        self.vectorizer = TfidfVectorizer(**kwargs)
        self.chunks = None
        self.vectors = None

    def init_chunks(self, chunks: list[str]):
        self.chunks = chunks
        fit_chunks = self.preprocess(chunks)
        self.vectors = self.vectorizer.fit_transform(fit_chunks)

    def rank(self, query: str, return_similarities: bool = False) -> list[str] | list[tuple[str, float]]:
        query_vector = self.vectorizer.transform([self.preprocess_chunk(query)])
        cosine_similarities = cosine_similarity(query_vector, self.vectors).flatten()
        if return_similarities:
            return [(self.chunks[i], cosine_similarities[i]) for i in cosine_similarities.argsort()[::-1]]
        return [x for _, x in sorted(zip(cosine_similarities, self.chunks), reverse=True)][:self.top_k]

    def preprocess(self, chunks: list[str]) -> list[str]:
        # remove punctuation, lowercase, and remove stopwords, if any
        return [self.preprocess_chunk(chunk) for chunk in chunks]

    @staticmethod
    def preprocess_chunk(chunk: str) -> str:
        # Handle contractions
        chunk = contractions.fix(chunk)

        # Remove punctuation
        chunk = chunk.translate(str.maketrans("", "", string.punctuation))

        # Lowercase
        chunk = chunk.lower()

        # Tokenize
        tokens = chunk.split()

        # Remove stopwords
        tokens = [word for word in tokens if word not in TfidfRanker.stop_words]

        # Stem
        tokens = [TfidfRanker.stemmer.stem(word) for word in tokens]

        # Remove non-alphabetic tokens
        tokens = [word for word in tokens if word.isalpha()]

        return ' '.join(tokens)

    def batch_rank(self, queries: list[str], batch_size: int = 100) -> list[list[str]]:
        query_vectors = self.vectorizer.transform([self.preprocess_chunk(query) for query in queries])
        cosine_similarities = cosine_similarity(query_vectors, self.vectors)
        return [[x for _, x in sorted(zip(cosine_similarities[i], self.chunks), reverse=True)][:self.top_k] for i in range(len(queries))]
