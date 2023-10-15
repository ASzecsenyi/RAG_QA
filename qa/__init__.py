from abc import ABC, abstractmethod


class QA(ABC):
    """
    Abstract class for QA models.
    """

    def __init__(self, model):
        self.model = model

    @abstractmethod
    def predict(self, question, context):
        """
        Predicts the answer to a question given a context.

        :param question: the question
        :type question: str
        :param context: the context
        :type context: str
        """
        raise NotImplementedError
