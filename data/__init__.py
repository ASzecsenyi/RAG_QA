from abc import ABC, abstractmethod
import json
import os
import pandas as pd


class Dataset(ABC):
    """
    Abstract class for datasets.

    stores document sets, questions over the document set, golden passages for each question, and golden answers for each question.
    """

    def __init__(self, name: str):
        """
        :param name: the dataset identifier - name used in paper it is released with, e.g. "hotpotqa"
        :type name: str
        """
        self.name = name
        self.document_sets = {}

    @abstractmethod
    def load(self):
        """
        Loads the dataset.
        """
        raise NotImplementedError

    def add_document_set(self, document_set: 'DocumentSet'):
        """
        Adds a document set to the dataset.

        :param document_set: the document set to add
        :type document_set: DocumentSet
        """
        self.document_sets[document_set.name] = document_set

    def get_document_set(self, document_set_name: str) -> 'DocumentSet':
        """
        Gets a document set from the dataset.

        :param document_set_name: the name of the document set to get
        :type document_set_name: str
        :return: the document set
        :rtype: DocumentSet
        """
        return self.document_sets[document_set_name]


class DocumentSet(ABC):
    """
    Abstract class for document sets.

    stores documents and their titles.
    """

    def __init__(self, name: str):
        """
        :param name: the document set identifier - name used in paper it is released with, e.g. "wikipedia"
        :type name: str
        """
        self.name = name

    @abstractmethod
    def load(self):
        """
        Loads the document set.
        """
        raise NotImplementedError

