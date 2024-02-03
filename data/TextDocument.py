from data import Document


class TextDocument(Document):
    """
    A document from the user upload
    """

    def __init__(self, document: str):
        """
        :param document: the content
        :type document: str
        """

        questions = [
            {
                "question": "",
                "ground_truths": [""],
            }
        ]

        super().__init__(document, None, questions)
