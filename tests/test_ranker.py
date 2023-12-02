import pytest
from sklearn.feature_extraction.text import TfidfVectorizer

from retrieval.Ranker import Ranker
from retrieval.Ranker.TfidfRanker import TfidfRanker  # replace with your actual module name


def test_ranker_init():
    r = TfidfRanker(10)
    assert r.chunks is None
    assert r.top_k == 10
    assert r.name == 'TfidfRanker_10_{}'


def test_tfidf_ranker_init():
    r = TfidfRanker(5)
    assert r.name == "TfidfRanker_5_{}"
    assert r.vectors is None
    assert isinstance(r.vectorizer, TfidfVectorizer)


def test_tfidf_ranker_init_chunks():
    r = TfidfRanker(5)
    r.init_chunks(['This is a chunk', 'This is another chunk', 'This is a third chunk'])
    assert r.chunks == ['This is a chunk', 'This is another chunk', 'This is a third chunk']
    assert r.vectors.shape == (3, 5)  # 3 chunks, 5 unique words - not 6 because 'a' is not considered a word


def test_tfidf_ranker_rank():
    r = TfidfRanker(5)
    r.init_chunks(['A bit more complex chunk', 'Once again, a chunk', 'A chunk'])
    assert r.rank('A chunk') == ['A chunk', 'Once again, a chunk', 'A bit more complex chunk']
