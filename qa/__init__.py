from abc import ABC, abstractmethod


class QA(ABC):
    """
    Abstract class for QA models.
    """

    def __init__(self, name: str):
        """
        Initializes the QA model.

        :param name: the name of the model
        :type name: str
        """

        self.name = name

    @abstractmethod
    def predict(self, question: str, chunks: list[str]) -> str:
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
