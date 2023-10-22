from abc import ABC, abstractmethod


class Chunker(ABC):
    def __init__(self, chunk_length: int, sliding_window_size: float = 0.0):
        """
        :param chunk_length: The length of the chunks. measurement depends on the chunker.
        :type chunk_length: int
        :param sliding_window_size: The size of the sliding window. Defines the overlap between chunks. Must be between 0.0 and 1.0, defaults to 0.0.
        :type sliding_window_size: float, optional
        """

        assert 0.0 <= sliding_window_size < 1.0, 'sliding_window_size must be between 0.0 and 1.0'

        self.chunk_length = chunk_length
        self.sliding_window_size = sliding_window_size

    @abstractmethod
    def chunk(self, document: str) -> list[str]:
        """
        Chunks a document into smaller pieces.

        :param document: The document
        :type document: str

        :return: The chunks
        """
        raise NotImplementedError
