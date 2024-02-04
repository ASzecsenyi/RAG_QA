from typing import Any

from data import Document


class TextDocument(Document):
    """
    A document from the user upload
    """

    def __init__(self, document: str, name: str = None, questions: list[dict[str, Any]] = None):
        """
        :param document: the content
        :type document: str
        """

        if questions is None:
            questions = [
                {
                    "question": "",
                    "ground_truths": [""],
                }
            ]

        super().__init__(document, name, questions)
