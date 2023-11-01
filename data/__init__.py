from abc import ABC
import json
from typing import Any
from uuid import uuid4


class Document(ABC):
    """
    Abstract class for datasets.

    stores document sets, questions over the document set, golden passages for each question, and golden answers for each question.
    """

    def __init__(self, document: str, name: str = None, questions: list[dict[str, Any]] = None):
        """
        :param name: the dataset identifier - name used in paper it is released with, e.g. "hotpot_qa"
        :type name: str, optional
        :param document: the document to get data from
        :type document: str
        :param questions: the evaluation data, defaults to None - uses format [{"question": "Whodunnit?", "answers": "The butler."}, ]
        :type questions: list[dict[str, Any]], optional
        """
        self.name = name
        if name is None:
            self.name = "_".join(document.split()[:5]) + f"_{uuid4()}"
        self.document: str = document
        self.questions: list[dict[str, Any]] = questions

        assert all("question" in question for question in self.questions), "questions must have a question field"

    @classmethod
    def from_json(cls, path: str, name: str = None) -> "Document":
        """
        Creates a dataset from a json file.

        :param path: the path to the json file
        :type path: str
        :param name: the name of the dataset, defaults to None - uses the name in the json file
        :type name: str, optional

        :return: the dataset
        :rtype: Document
        """
        with open(path, "r") as f:
            data = json.load(f)

        if name is None:
            name = data["name"]

        return cls(name, data["document"], data["questions"])

    def save(self, path: str = None):
        """
        Saves the dataset to a json file.

        :param path: the path to save the json file to
        :type path: str
        """

        if path is None:
            path = f"../data/files/{self.name}.json"

        with open(path, "w") as f:
            json.dump(self.__dict__, f, indent=4)

