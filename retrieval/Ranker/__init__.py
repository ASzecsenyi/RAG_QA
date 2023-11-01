from abc import ABC, abstractmethod
from uuid import uuid4


class Ranker(ABC):
    def __init__(self, top_k: int, name: str = None):
        """
        :param top_k: The number of chunks to return
        :type top_k: int
        :param name: The name of the ranker, defaults to None - uses the class name and a uuid
        :type name: str, optional
        """
        self.chunks = None
        self.top_k = top_k
        self.name = name
        if name is None:
            self.name = self.__class__.__name__ + f"_{top_k}_{uuid4()}"

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
