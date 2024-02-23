import os
import time

import numpy as np
import requests
import torch

from retrieval.Ranker import Ranker
from retrieval.Ranker.CrossEncodingRanker import CrossEncodingRanker


class GuessSimilarityRanker(Ranker):
    def __init__(self, top_k: int, num_of_paraphrases: int = 5, ranker: Ranker = CrossEncodingRanker(top_k=5), api_key: str = None, name=None, **kwargs):
        """
        :param top_k: The number of chunks to return
        :type top_k: int

        :param kwargs: Keyword arguments for the TfidfVectorizer
        :type kwargs: dict
        """
        super().__init__(top_k, name=name,)
        self.api_key = api_key
        if api_key is None:
            try:
                self.api_key = os.environ["HUGGINGFACE_API_KEY"]
            except KeyError:
                raise KeyError(
                    "HUGGINGFACE_API_KEY environment variable not found. Please set it or pass it as an argument."
                )
        if name is None:
            self.name += f"_{ranker.name}" + f"_{kwargs}" if kwargs else ""
        self.num_of_paraphrases = num_of_paraphrases

        device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model = ranker

    def init_chunks(self, chunks: list[str]):
        self.chunks: list[str] = chunks
        self.model.init_chunks(chunks)

    def rank(self, query: str, return_similarities: bool = False) -> list[str] | list[tuple[str, float]]:
        answers: list[str] = self.get_guesses(query)
        # Get vectors for each answer
        scores = []
        for answer in answers:
            scores.append(self.model.rank(answer, return_similarities=True))
        # calculate the mean similarity for each chunk
        ranks_mean = np.mean([np.array([score[1] for score in chunk_scores]) for chunk_scores in scores], axis=0)
        # Sort by mean similarity
        sorted_indices = np.argsort(ranks_mean)[::-1]
        # Return the chunks in order of similarity
        if return_similarities:
            return [(self.chunks[i], ranks_mean[i]) for i in sorted_indices], answers
        return [self.chunks[i] for i in sorted_indices[:self.top_k]], answers

    def get_guesses(self, query: str) -> list[str]:
        inputs = (f"<s>[INST] You will receive a question. "
                  f"Come up with {self.num_of_paraphrases} different ways the sentence that contains the answer might look like. "
                  f"You don't have to come up with an answer, and you can replace it with the word '[MASK]' in each of the sentences."
                  f"Separate the sentences with a semicolon. "
                  f"Example: "
                  f"Q: Who was driving th vehicle?"
                  f"A: MASK was driving the vehicle; The person behind the wheel was [MASK]; The person in the driver's seat was [MASK]; The car was driven by [MASK]; They were [MASK]'s passengers;"
                  f"[/INST] "
                  f"Thank you, I will do accordingly, as you have instructed. "
                  f"[INST] {query} [/INST]")

        try:
            response = requests.post(
                "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.1",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={"inputs": inputs,
                      "parameters": {"max_new_tokens": 100}})
        except requests.exceptions.ConnectionError:
            print("Paused due to lost connection. Press enter to continue.")
            return self.get_guesses(query)
        try:
            response.raise_for_status()
        except requests.exceptions.HTTPError:
            # if too many requests are made, the server will return a 503 status code
            print("Too many requests. Waiting 10 seconds.")
            time.sleep(10)
            return self.get_guesses(query)

        response_str = response.json()[0]["generated_text"].split("[/INST]")[-1].strip()

        r_list = response_str.split(";\n")

        if len(r_list) == 1:
            r_list = r_list[0].split("\n")
        if len(r_list) == 1:
            r_list = r_list[0].split(";")

        if len(r_list) < self.num_of_paraphrases:
            r_list += ["[UNKNOWN]"] * (self.num_of_paraphrases - len(r_list))

        print(r_list)

        return r_list

    def batch_rank(self, queries: list[str], batch_size: int = 100) -> list[list[str]]:
        raise NotImplementedError
