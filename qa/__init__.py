from abc import ABC, abstractmethod

from typing import List


class QA(ABC):
    """
    Abstract class for QA models.
    """

    def __init__(self, name: str = None):
        """
        Initializes the QA model.

        :param name: The name of the QA model, defaults to None - uses the class name
        :type name: str, optional
        """

        self.name = name
        if name is None:
            self.name = self.__class__.__name__

    @abstractmethod
    def predict(self, question: str, chunks: List[str]) -> str:
        """
        Predicts the answer to a question given a context.

        :param question: the question
        :type question: str
        :param chunks: the context
        :type chunks: list[str]

        :return: the answer
        :rtype: str
        """
        raise NotImplementedError
