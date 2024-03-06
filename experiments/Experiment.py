import json
import os
import time
from datetime import datetime
from uuid import uuid4

import numpy as np
from ragas import evaluate
from datasets import Dataset
from typing import Union, Any
from rouge_score import rouge_scorer
from tqdm import tqdm

from data import Document
from retrieval import Chunker, Ranker
from qa import QA
from retrieval.Ranker.GuessSimilarityRanker import GuessSimilarityRanker
from retrieval.Ranker.PromptRanker import PromptRanker


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

    return evaluate(dataset=results_dataset)


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
        if name is None:
            self.name = "experiment-" + str(uuid4())
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

    def run(self, get_ground_ranks: bool = False) -> dict[str, dict[Any, Any]]:
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
                times = results["times"]
            except FileNotFoundError:
                print("No results found, running experiment")

        start_time = time.time()
        total_num_of_results = len(self.qa) * len(self.chunker) * len(self.ranker) * sum([len(dataset.questions) for dataset in self.dataset])
        num_of_processed_results = 0

        for dataset in self.dataset:
            if hasattr(dataset, "paragraphs"):
                paragraphs = dataset.paragraphs
            for chunker in self.chunker:
                if isinstance(self.results, dict) and all(result_setup in self.results for result_setup in [f"{chunker.name}_{ranker.name}_{qa.name}_{dataset.name}" for ranker in self.ranker for qa in self.qa]):
                    print(f"Results for {dataset.name}, {chunker.name} already found, skipping")
                    continue
                chunks = self.r(chunker.chunk, f"Chunking with {chunker.name}", times, document=dataset.document)

                for ranker in self.ranker:
                    if isinstance(self.results, dict) and all(result_setup in self.results for result_setup in [f"{chunker.name}_{ranker.name}_{qa.name}_{dataset.name}" for qa in self.qa]):
                        print(f"Results for {dataset.name}, {chunker.name}, {ranker.name} already found, skipping")
                        continue
                    if isinstance(ranker, PromptRanker) and hasattr(dataset, "paragraphs"):
                        self.r(ranker.init_chunks, f"Initialising ranker {ranker.name} with chunks", times,
                               chunks=chunks, paragraphs=paragraphs)
                    else:
                        self.r(ranker.init_chunks, f"Initialising ranker {ranker.name} with chunks", times, chunks=chunks)

                    results.update({f"{chunker.name}_{ranker.name}_{qa.name}_{dataset.name}": [] for qa in self.qa})
                    if num_of_processed_results > 0:
                        # calculate time left estimate
                        time_left = (total_num_of_results - num_of_processed_results) * (time.time() - start_time) / num_of_processed_results
                        print(f"{datetime.now().strftime('%H:%M:%S')}: Estimated time left: {datetime.fromtimestamp(time_left).strftime('%H:%M:%S')}")
                    for question in tqdm(dataset.questions, desc=f"Running experiment on {dataset.name} with {chunker.name} and {ranker.name}"):
                        if not get_ground_ranks:
                            contexts = self.r(ranker.rank, f"Ranking question with ranker {ranker.name}", times, query=question["question"], silenced=True)
                            if isinstance(ranker, GuessSimilarityRanker) or isinstance(ranker, PromptRanker):
                                contexts, guesses = contexts
                        else:
                            contexts = self.r(ranker.rank, f"Ranking question with ranker {ranker.name}", times, query=question["question"], silenced=True, return_similarities=True)
                            if isinstance(ranker, GuessSimilarityRanker) or isinstance(ranker, PromptRanker):
                                contexts, guesses = contexts
                            ground_rank = -1
                            ground_distance = -1
                            for i, context in enumerate(contexts):
                                chunk, distance = context
                                if any(ground_truth in chunk for ground_truth in question["ground_truths"]):
                                    ground_rank = i
                                    ground_distance = float(distance)
                                    break

                            # normalise ground rank
                            ground_rank /= len(contexts)
                            # normalise ground distance
                            ground_distance = (ground_distance - min([distance for _, distance in contexts])) / (max([distance for _, distance in contexts]) - min([distance for _, distance in contexts]))
                            # replace NaNs with 0
                            ground_distance = 0 if np.isnan(ground_distance) else ground_distance
                            contexts = [context[0] for context in contexts[:ranker.top_k]]

                        for qa in self.qa:

                            # if the question does not already have an answer in results, predict one
                            if not any(result["question"] == question["question"] for result in results[f"{chunker.name}_{ranker.name}_{qa.name}_{dataset.name}"]):
                                answer = self.r(qa.predict, f"Generating response with {qa.name}", times, question=question["question"], chunks=contexts, silenced=True)

                                result = {
                                    "question": question["question"],
                                    "answer": answer,
                                    "ground_truths": question["ground_truths"],
                                    "contexts": contexts
                                }
                                if isinstance(ranker, GuessSimilarityRanker) or isinstance(ranker, PromptRanker):
                                    result["guesses"] = guesses
                                if get_ground_ranks:
                                    result["ground_rank"] = ground_rank
                                    result["ground_distance"] = ground_distance

                                results[f"{chunker.name}_{ranker.name}_{qa.name}_{dataset.name}"].append(result)
                                num_of_processed_results += 1
                            else:
                                # if self.verbose:
                                print(f"Question {question['question']} already answered")
                            results["times"] = times
                            self.results = results

                            self.save_results()

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
        abs_paths = [os.path.abspath(f"../data/experiments/{file}") for file in files]
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

        if "evaluations" in self.results:
            return self.results["evaluations"]

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
            evaluations[result_setup]["ground_rank"] = sum(result.get("ground_rank", 0) for result in results) / len(results)
            evaluations[result_setup]["ground_distance"] = sum(result.get("ground_distance", 0) for result in results) / len(results)

        overall = {}

        # average all evaluations that are the same setup except for the document
        for result_setup, evaluation in evaluations.items():
            if result_setup == "times":
                continue

            setup = "_".join(result_setup.split("_")[:-2])
            if setup not in overall:
                overall[setup] = {key: 0 for key in evaluation.keys()}
                overall[setup]["retrieval"] = 0
                overall[setup]["ground_rank"] = 0
                overall[setup]["ground_distance"] = 0
            for key, value in evaluation.items():
                overall[setup][key] += value/len(self.dataset)

        self.results["evaluations"] = evaluations
        self.results["overall"] = overall

        return evaluations

    def __repr__(self):
        return f"Experiment(name={self.name}, description={self.description}, dataset={self.dataset}, chunker={self.chunker}, ranker={self.ranker}, qa={self.qa})"
