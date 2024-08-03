from typing import Any, List, Dict

from data import Document


class TextDocument(Document):
    """
    A document from the user upload
    """

    def __init__(self, document: str, name: str = None, questions: List[Dict[str, Any]] = None):
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
