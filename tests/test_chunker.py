import pytest
from retrieval.Chunker import Chunker
from retrieval.Chunker.CharChunker import CharChunker
from retrieval.Chunker.SentChunker import SentChunker


def test_chunker_init():
    with pytest.raises(AssertionError):
        CharChunker(-1, -1)

    c = CharChunker(1, 0.5)
    assert c.chunk_length == 1
    assert c.sliding_window_size == 0.5
    assert c.name == 'CharChunker_1_0.5'


def test_char_chunker_init():
    c = CharChunker(10, 0.5)
    assert c.chunk_length == 10
    assert c.sliding_window_size == 0.5
    assert c.name == 'CharChunker_10_0.5'


def test_char_chunker_chunk():
    c = CharChunker(3, 0.5)
    document = 'abcdefgh'
    chunks = c.chunk(document)

    assert chunks == ['abc', 'cde', 'efg', 'gh']

    c = CharChunker(3, 0.0)
    chunks = c.chunk(document)
    assert chunks == ['abc', 'def', 'gh']


def test_sent_chunker_init():
    c = SentChunker(10, 0.5)
    assert c.chunk_length == 10
    assert c.sliding_window_size == 0.5
    assert c.name == 'SentChunker_10_0.5'


def test_sent_chunker_chunk():
    document = 'Aa. Bb. Cc. dd. Ee. ff. Gg. Hh.'

    c = SentChunker(3, 0.0)
    chunks = c.chunk(document)
    assert chunks == ['Aa. Bb. Cc.', 'dd. Ee. ff.', 'Gg. Hh.'], f"window size 0.0"

    c = SentChunker(3, 0.5)
    chunks = c.chunk(document)
    assert chunks == ['Aa. Bb. Cc.', 'Cc. dd. Ee.', 'Ee. ff. Gg.', 'Gg. Hh.'], f"window size 0.5"


