import json
import os
import time

from ragas import evaluate
from datasets import Dataset
from typing import Union, Dict, Any
from rouge_score import rouge_scorer

from data import Document
from retrieval import Chunker, Ranker
from qa import QA


def evaluate_rouge_score(answers, ground_truths):
    """
    Evaluates the answers with the ROGUE score.

    :param answers: The predicted answers
    :type answers: list[str]
    :param ground_truths: The ground truths
    :type ground_truths: list[str]

    :return: The ROGUE score
    :rtype: float
    """

    scorer = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=True)
    scores = [scorer.score(prediction=answer, target=' '.join(ground_truth)) for answer, ground_truth in zip(answers, ground_truths)]
    return scores


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

        if not isinstance(self.dataset, list):
            self.dataset = [self.dataset]

        if not isinstance(self.chunker, list):
            self.chunker = [self.chunker]

        if not isinstance(self.ranker, list):
            self.ranker = [self.ranker]

        if not isinstance(self.qa, list):
            self.qa = [self.qa]

        self.results = None
        self.verbose = False

    def r(self, function, text, times, silenced=False, **kwargs):
        if self.verbose and not silenced:
            print(f"{text}")

        start_time = time.time()

        result = function(**kwargs)

        if text not in times:
            times[text] = []
        times[text].append(time.time() - start_time)

        return result

    def run(self) -> dict[str, dict[Any, Any]]:
        """
        Runs the experiment(s).

        If chunker, ranker, or qa is a list, the experiment runs all combinations of chunkers, rankers, and qa models.

        :return: The results of the experiment
        :rtype: dict[str, list[dict[str, str]]]
        """

        results = {}

        times = {}
        try:
            self.load_results()
        except FileNotFoundError:
            pass

        for dataset in self.dataset:
            for chunker in self.chunker:
                chunks = self.r(chunker.chunk, "Chunking", times, document=dataset.document)

                for ranker in self.ranker:
                    self.r(ranker.init_chunks, "Initialising ranker", times, chunks=chunks)

                    results.update({f"{chunker.name}_{ranker.name}_{qa.name}_{dataset.name}": [] for qa in self.qa})
                    for question in dataset.questions:
                        chunks = self.r(ranker.rank, "Ranking", times, query=question["question"], silenced=True)

                        for qa in self.qa:
                            # if the question does not already have an answer in results, predict one
                            if not any(result["question"] == question["question"] for result in results[f"{chunker.name}_{ranker.name}_{qa.name}_{dataset.name}"]):
                                answer = self.r(qa.predict, "Qa", times, question=question["question"], chunks=chunks, silenced=True)

                                results[f"{chunker.name}_{ranker.name}_{qa.name}_{dataset.name}"].append(
                                    {"question": question["question"], "answer": answer, "ground_truths": question["ground_truths"], "contexts": chunks}
                                )
                            self.results = results

                            self.save_results()

        results["times"] = times

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

    def load_results(self, path: str = None):
        """
        Loads the experiment results from a json file.

        :param path: The path to load the json file from
        :type path: str
        """

        if path is None:
            assert os.path.exists(f"../data"), "Experiment must be run before loading results"
            assert os.path.exists(f"../data/files"), "Experiment must be run before loading results"
            path = f"../data/files/{self.name}.json"

        with open(path, "r") as f:
            saved_results = json.load(f)

        self.name = saved_results["name"]
        self.description = saved_results["description"]
        self.results = saved_results["results"]

    def evaluate_with_ragas(self):
        assert self.results is not None, "Experiment must be run before evaluation"

        evalutions = {}

        for result_setup, results in self.results.items():
            if result_setup == "times":
                continue

            # prepare results dataset in the format
            # Dataset({
            #     features: ['question', 'contexts', 'answer', 'ground_truths'],
            #     num_rows: 25
            # })
            #
            # where `answer` is the predicted answer

            # get all questions and answers
            questions = [result["question"] for result in results]
            answers = [result["answer"] for result in results]

            # get all contexts
            contexts = [result["contexts"] for result in results]

            # get all ground truths
            ground_truths = [result["ground_truths"] for result in results]

            results_dataset = Dataset.from_dict(
                {
                    "question": questions,
                    "contexts": contexts,
                    "answer": answers,
                    "ground_truths": ground_truths
                }
            )

            # evaluate results with ragas
            evaluation = evaluate(results_dataset)

            evalutions[result_setup] = evaluation

        return evalutions

    def evaluate_with_rouge_score(self):
        assert self.results is not None, "Experiment must be run before evaluation"

        evalutions = {}

        for result_setup, results in self.results.items():
            if result_setup == "times":
                continue

            # get all questions and answers
            answers = [result["answer"] for result in results]

            # get all ground truths
            ground_truths = [result["ground_truths"] for result in results]

            # evaluate results with rogue score
            evaluation = evaluate_rouge_score(answers, ground_truths)

            mean_evaluation = {}
            for key in evaluation[0].keys():
                mean_evaluation[key] = sum([e[key] for e in evaluation]) / len(evaluation)

            evalutions[result_setup] = evaluation

        return evalutions









