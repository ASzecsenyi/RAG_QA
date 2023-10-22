from abc import ABC, abstractmethod


class Ranker(ABC):
    def __init__(self, top_k: int):
        """
        :param top_k: The number of chunks to return
        :type top_k: int
        """
        self.chunks = None
        self.top_k = top_k

    @abstractmethod
    def init_chunks(self, chunks: list[str]):
        """
        Initialises the ranker with a list of chunks.
        :param chunks: The chunks
        :type chunks: list[str]
        """
        raise NotImplementedError

    @abstractmethod
    def rank(self, query: str) -> list[str]:
        """
        Ranks the chunks based on a query.
        :param query: The query
        :type query: str

        :return: The top-k chunks ordered by relevance (descending)
        """
        raise NotImplementedError
