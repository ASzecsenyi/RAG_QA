import nltk
from typing import List

from retrieval.Chunker import Chunker

# if needed, download the punkt tokenizer
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')


class WordChunker(Chunker):
    def __init__(self, chunk_length: int, sliding_window_size: float = 0.0, name: str = None):
        """
        :param chunk_length: The length of the chunks, measured in sentences.
        :type chunk_length: int
        :param sliding_window_size: The size of the sliding window. Defines the overlap between chunks, rounded down. Must be between 0.0 and 1.0, defaults to 0.0.
        :type sliding_window_size: float, optional
        """
        super().__init__(chunk_length, sliding_window_size, name)

    def chunk(self, document: str) -> List[str]:
        words = nltk.word_tokenize(document)

        if len(words) < self.chunk_length:
            return [' '.join(words)]

        return [' '.join(words[i:i + self.chunk_length]) for i in
                range(
                    0,
                    len(words),
                    self.chunk_length - int(self.chunk_length * self.sliding_window_size)
                )]
