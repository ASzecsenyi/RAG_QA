from abc import ABC, abstractmethod
from data import DocumentSet


class Retrieval(ABC):
    """
    Abstract class for retrieval models.
    """

    def __init__(self, name: str):
        """
        :param name: the model identifier - name used in paper it is released with, e.g. "tfidf"
        :type name: str
        """
        self.name = name

    @abstractmethod
    def predict(self, query: str, documents: DocumentSet):
        """
        Predicts the answer to a question given a context.

        :param query: the query
        :type query: str
        :param documents: the documents
        :type documents: DocumentSet
        """
        raise NotImplementedError
