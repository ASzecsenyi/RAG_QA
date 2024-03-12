from abc import ABC
import json
from typing import Any


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
        :param questions: the evaluation data, defaults to None - uses format [{"question": "Whodunnit?", "ground_truths": ["The butler did it."]}]
        :type questions: list[dict[str, Any]], optional
        """
        self.name = name
        if name is None:
            self.name = "_".join(document.split()[:5])
        self.document: str = document
        self.questions: list[dict[str, Any]] = questions

        assert isinstance(self.document, str), "document must be a string"
        assert isinstance(self.questions, list), "questions must be a list"
        assert all(isinstance(question, dict) for question in self.questions), "questions must be a list of dictionaries"

        assert all("question" in question for question in self.questions), "questions must have a question field"
        assert all(isinstance(question["question"], str) for question in self.questions), "question fields must be strings"
        assert all("ground_truths" in question for question in self.questions), "questions must have a ground_truths field"
        assert all(isinstance(question["ground_truths"], list) for question in self.questions), "ground_truths fields must be lists"
        assert all(all(isinstance(ground_truth, str) for ground_truth in question["ground_truths"]) for question in self.questions), f"ground_truths fields must be lists of strings, got {type(self.questions[0]['ground_truths'][0])}"

    def __repr__(self):
        return f"Document(name={self.name}, document={self.document[:10]}, questions={self.questions})"

    def __str__(self):
        return f"Document(name={self.name}, document={self.document[:10]}, questions={self.questions})"

    def __len__(self):
        return len(self.document)

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

        return cls(name=name, document=data["document"], questions=data["questions"])

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
