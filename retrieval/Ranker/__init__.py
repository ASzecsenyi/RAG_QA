import random
from abc import ABC, abstractmethod


class Ranker(ABC):
    def __init__(self, top_k: int, name: str = None):
        """
        :param top_k: The number of chunks to return
        :type top_k: int
        :param name: The name of the ranker, defaults to None - uses the class name
        :type name: str, optional
        """
        self.chunks = None
        self.top_k = top_k
        self.name = name
        if name is None:
            self.name = self.__class__.__name__ + f"_{top_k}"

    @abstractmethod
    def init_chunks(self, chunks: list[str]):
        """
        Initialises the ranker with a list of chunks.
        :param chunks: The chunks
        :type chunks: list[str]
        """
        raise NotImplementedError

    @abstractmethod
    def rank(self, query: str, return_similarities: bool = False) -> list[str] | list[tuple[str, float]]:
        """
        Ranks the chunks based on a query.
        :param query: The query
        :type query: str
        :param return_similarities: Whether to return the distances/similarities, defaults to False
        :type return_similarities: bool, optional

        :return: The top-k chunks ordered by relevance (descending)
        """
        raise NotImplementedError

    @abstractmethod
    def batch_rank(self, queries: list[str], batch_size: int = 100) -> list[list[str]]:
        """
        Ranks the chunks based on a list of queries.
        :param queries: The queries
        :type queries: list[str]
        :param batch_size: The batch size
        :type batch_size: int

        :return: The top-k chunks ordered by relevance (descending) for each query
        """
        raise NotImplementedError


class RandomRanker(Ranker):
    def __init__(self, top_k: int, name=None):
        """
        :param top_k: The number of chunks to return
        :type top_k: int
        """
        super().__init__(top_k, name)

    def init_chunks(self, chunks: list[str]):
        self.chunks = chunks

    def rank(self, query: str, return_similarities: bool = False) -> list[str] | list[tuple[str, float]]:
        if return_similarities:
            order = random.sample(range(len(self.chunks)), len(self.chunks))
            return [(self.chunks[i], 1 - i / len(self.chunks)) for i in order]
        return random.sample(self.chunks, self.top_k)

    def batch_rank(self, queries: list[str], batch_size: int = 100) -> list[list[str]]:
        return [random.sample(self.chunks, self.top_k) for _ in range(len(queries))]
