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


try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
# if not nltk.data.find('corpora/wordnet'):
#    nltk.download('wordnet')


class TfidfRanker(Ranker):
    stop_words = set(stopwords.words('english'))
    # lemmatizer = WordNetLemmatizer()
    stemmer = PorterStemmer()

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
        self.chunks = None
        self.vectors = None

    def init_chunks(self, chunks: list[str]):
        self.chunks = chunks
        fit_chunks = self.preprocess(chunks)
        self.vectors = self.vectorizer.fit_transform(fit_chunks)

    def rank(self, query: str) -> list[str]:
        query_vector = self.vectorizer.transform([self.preprocess_chunk(query)])
        cosine_similarities = cosine_similarity(query_vector, self.vectors).flatten()
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

        # Lemmatize
        # tokens = [TfidfRanker.lemmatizer.lemmatize(word) for word in tokens]

        # Stem
        tokens = [TfidfRanker.stemmer.stem(word) for word in tokens]

        # Remove non-alphabetic tokens
        tokens = [word for word in tokens if word.isalpha()]

        return ' '.join(tokens)

