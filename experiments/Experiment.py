import json
from typing import Union
from data import Document
from retrieval import Chunker, Ranker
from qa import QA


class Experiment:
    """Experiment class for running experiments on a dataset with a given pipeline or set of pipelines."""
    def __init__(
            self,
            name: str,
            description: str,
            dataset: Union[Document, list[Document]],
            chunker: Union[Chunker, list[Chunker]],
            ranker: Union[Ranker, list[Ranker]],
            qa: Union[QA, list[QA]]
    ):
        """
        :param name: The name of the experiment
        :type name: str
        :param description: The description of the experiment
        :type description: str
        :param dataset: The dataset to run the experiment on
        :type dataset: Document
        :param chunker: The chunker or chunkers to use
        :type chunker: Union[Chunker, list[Chunker]]
        :param ranker: The ranker or rankers to use
        :type ranker: Union[Ranker, list[Ranker]]
        :param qa: The QA model or models to use
        :type qa: Union[QA, list[QA]]
        """
        self.name = name
        self.description = description
        self.dataset = dataset
        self.chunker = chunker
        self.ranker = ranker
        self.qa = qa

        self.results = None

    def run(self) -> dict[str, list[dict[str, str]]]:
        """
        Runs the experiment(s).

        If chunker, ranker, or qa is a list, the experiment runs all combinations of chunkers, rankers, and qa models.

        :return: The results of the experiment
        :rtype: dict[str, list[dict[str, str]]]
        """

        if not isinstance(self.chunker, list):
            self.chunker = [self.chunker]

        if not isinstance(self.ranker, list):
            self.ranker = [self.ranker]

        if not isinstance(self.qa, list):
            self.qa = [self.qa]

        if not isinstance(self.dataset, list):
            self.dataset = [self.dataset]

        results = {}
        for dataset in self.dataset:
            for chunker in self.chunker:
                for ranker in self.ranker:
                    results.update({f"{chunker.name}_{ranker.name}_{qa.name}_{dataset.name}": [] for qa in self.qa})

                    chunks = chunker.chunk(dataset.document)
                    ranker.init_chunks(chunks)

                    for question in dataset.questions:
                        chunks = ranker.rank(question["question"])
                        for qa in self.qa:
                            answer = qa.predict(question["question"], chunks)
                            results[f"{chunker.name}_{ranker.name}_{qa.name}_{dataset.name}"].append(
                                {"question": question["question"], "answer": answer}
                            )

        self.results = results
        return results

    def save_results(self, path: str = None):
        """
        Saves the experiment results to a json file.

        :param path: The path to save the json file to
        :type path: str
        """

        if path is None:
            path = f"../data/files/{self.name}.json"

        saved_results = {"name": self.name, "description": self.description, "results": self.results}

        with open(path, "w") as f:
            json.dump(saved_results, f, indent=4)
