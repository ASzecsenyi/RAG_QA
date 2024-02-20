import json
import os
import time
from datetime import datetime

from ragas import evaluate
from datasets import Dataset
from typing import Union, Any
from rouge_score import rouge_scorer
from tqdm import tqdm

from data import Document
from retrieval import Chunker, Ranker
from qa import QA
from retrieval.Ranker.GuessSimilarityRanker import GuessSimilarityRanker


def evaluate_rouge_score(answers: list[str], ground_truths: list[str]) -> dict[str, list[float]]:
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

    # print(scores)

    final_score = {
        'fmeasure': [score['rouge1'].fmeasure for score in scores],
        'precision': [score['rouge1'].precision for score in scores],
        'recall': [score['rouge1'].recall for score in scores]
    }

    return final_score


def evaluate_ragas_score(results_dataset: Dataset):
    """
    Evaluates the results with ragas.

    :param results_dataset: The results dataset
    :type results_dataset: Dataset

    :return: The evaluation
    :rtype: dict[str, dict[str, float]]
    """

    return evaluate(results_dataset)


class Experiment:
    """Experiment class for running experiments on a dataset with a given pipeline or set of pipelines."""
    def __init__(
            self,
            name: str,
            description: str,
            dataset: Union[Document, list[Document]],
            chunker: Union[Chunker, list[Chunker]],
            ranker: Union[Ranker, list[Ranker]],
            qa: Union[QA, list[QA]],
            autoload: bool = True
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
        self.evaluation = None
        self.name = name
        self.description = description

        self.dataset = dataset
        self.chunker = chunker
        self.ranker = ranker
        self.qa = qa

        self.autoload = autoload

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
        # if self.verbose and not silenced:
        #     print(f"{text}")

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
        if self.autoload:
            try:
                self.load_results(get_name=False)
                results = self.results
            except FileNotFoundError:
                print("No results found, running experiment")

        for dataset in self.dataset:
            for chunker in self.chunker:
                if isinstance(self.results, dict) and all(result_setup in self.results for result_setup in [f"{chunker.name}_{ranker.name}_{qa.name}_{dataset.name}" for ranker in self.ranker for qa in self.qa]):
                    print(f"Results for {dataset.name}, {chunker.name} already found, skipping")
                    continue
                chunks = self.r(chunker.chunk, f"Chunking data {dataset.name} with {chunker.name}", times, document=dataset.document)

                for ranker in self.ranker:
                    self.r(ranker.init_chunks, f"Initialising ranker {ranker.name} with chunks", times, chunks=chunks)

                    results.update({f"{chunker.name}_{ranker.name}_{qa.name}_{dataset.name}": [] for qa in self.qa})
                    for question in tqdm(dataset.questions, desc=f"Running experiment on {dataset.name} with {chunker.name} and {ranker.name}"):
                        contexts = self.r(ranker.rank, f"Ranking question {question}", times, query=question["question"], silenced=True)
                        if isinstance(ranker, GuessSimilarityRanker):
                            contexts, guesses = contexts
                        for qa in self.qa:
                            # if the question does not already have an answer in results, predict one
                            if not any(result["question"] == question["question"] for result in results[f"{chunker.name}_{ranker.name}_{qa.name}_{dataset.name}"]):
                                answer = self.r(qa.predict, "Qa", times, question=question["question"], chunks=contexts, silenced=True)

                                result = {
                                    "question": question["question"],
                                    "answer": answer,
                                    "ground_truths": question["ground_truths"],
                                    "contexts": contexts
                                }
                                if isinstance(ranker, GuessSimilarityRanker):
                                    result["guesses"] = guesses

                                results[f"{chunker.name}_{ranker.name}_{qa.name}_{dataset.name}"].append(result)
                            else:
                                # if self.verbose:
                                print(f"Question {question['question']} already answered")
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
            path = f"../data/experiments/{self.name}-{datetime.now().strftime('%m-%d-%H-%M-%S')}.json"
            # create the directory if it does not exist
            if not os.path.exists(f"../data/experiments"):
                os.makedirs(f"../data/experiments")

        saved_results = {"name": self.name, "description": self.description, "results": self.results}

        with open(path, "w") as f:
            json.dump(saved_results, f, indent=4)

        # print(f"Results saved to {path}")

        # if more than 5 files with the same name, delete the oldest
        files = os.listdir(f"../data/experiments")
        files.sort(key=lambda x: os.path.getmtime(f"../data/experiments/{x}"))
        files = [file for file in files if file.startswith(self.name)]
        if len(files) > 5:
            # remove the oldest file with the same name
            os.remove(f"../data/experiments/{files[0]}")

    def load_results(self, path: str = None, get_name: bool = True):
        """
        Loads the experiment results from a json file.

        :param path: The path to load the json file from
        :type path: str
        """

        if path is None:
            if not os.path.exists(f"../data"):
                raise FileNotFoundError("No data directory found")
            if not os.path.exists(f"../data/experiments"):
                raise FileNotFoundError("No experiments directory found")
            # find the latest experiment file
            files = os.listdir(f"../data/experiments")
            if get_name:
                files.sort(key=lambda x: os.path.getmtime(f"../data/experiments/{x}"))

                if len(files) == 0:
                    raise FileNotFoundError("No experiment files found")
                path = f"../data/experiments/{files[-1]}"
            else:
                # get the latest experiment file that has the same name
                files = [file for file in files if file.startswith(self.name)]
                files.sort(key=lambda x: os.path.getmtime(f"../data/experiments/{x}"))
                if len(files) == 0:
                    raise FileNotFoundError("No experiment files found")
                path = f"../data/experiments/{files[-1]}"

        with open(path, "r") as f:
            saved_results = json.load(f)

        if get_name:
            self.name = saved_results["name"]
        self.description = saved_results["description"]
        self.results = saved_results["results"]

        print(f"Results loaded from {path}")

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
            evaluation = evaluate_ragas_score(results_dataset)

            evalutions[result_setup] = evaluation

        return evalutions

    def evaluate_with_rouge_score(self):
        assert self.results is not None, "Experiment must be run before evaluation"

        evaluations = {}

        for result_setup, results in self.results.items():
            if result_setup == "times":
                continue

            # get all questions and answers
            answers = [result["answer"] for result in results]

            # get all ground truths
            ground_truths = [result["ground_truths"] for result in results]

            # evaluate results with rogue score
            evaluation = evaluate_rouge_score(answers, ground_truths)

            # write the evaluation to self.results
            for key, value in evaluation.items():
                for i, result in enumerate(results):
                    if key not in result:
                        result[key] = value[i]
                        # if given context contains ground truth, set retrieval to 1
                        result["retrieval"] = 0
                        for ground_truth in result["ground_truths"]:
                            if any(ground_truth in context for context in result["contexts"]):
                                result["retrieval"] = 1
                                break



            evaluations[result_setup] = {key: sum(value) / len(value) for key, value in evaluation.items()}
            evaluations[result_setup]["retrieval"] = sum(result["retrieval"] for result in results) / len(results)

        overall = {}

        # average all evaluations that are the same setup except for the document
        for result_setup, evaluation in evaluations.items():
            if result_setup == "times":
                continue

            setup = "_".join(result_setup.split("_")[:-2])
            if setup not in overall:
                overall[setup] = {key: 0 for key in evaluation.keys()}
                overall[setup]["retrieval"] = 0
            for key, value in evaluation.items():
                overall[setup][key] += value/len(self.dataset)

        self.results["evaluations"] = evaluations
        self.results["overall"] = overall

        return evaluations

    def __repr__(self):
        return f"Experiment(name={self.name}, description={self.description}, dataset={self.dataset}, chunker={self.chunker}, ranker={self.ranker}, qa={self.qa})"
