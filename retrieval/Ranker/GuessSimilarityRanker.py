import os
import time

import faiss
import numpy as np
import requests
import torch
from sentence_transformers import SentenceTransformer

from retrieval.Ranker import Ranker


class GuessSimilarityRanker(Ranker):
    def __init__(self, top_k: int, num_of_paraphrases: int = 5, api_key: str = None, name=None, **kwargs):
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
        self.index = None
        if kwargs and name is None:
            self.name += f"_{kwargs}"
        self.vectors = None
        self.num_of_paraphrases = num_of_paraphrases

        device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model = SentenceTransformer('paraphrase-MiniLM-L6-v2', device=device)

    def init_chunks(self, chunks: list[str]):
        self.chunks: list[str] = chunks
        vectors = self.model.encode(self.chunks)

        embedding_size = len(vectors[0])

        chunks_arr: np.ndarray = np.vstack(vectors, dtype="float32")

        faiss.normalize_L2(chunks_arr)

        # #### FAISS ####
        self.index = faiss.index_factory(embedding_size, "Flat", faiss.METRIC_INNER_PRODUCT)
        self.index.add(chunks_arr)

    def rank(self, query: str) -> list[str]:
        answers: list[str] = self.get_guesses(query)
        # print(answers)
        query_vectors = self.model.encode(answers)
        query_arr: np.ndarray = np.vstack(query_vectors, dtype="float32")
        if query_arr.shape[1] != self.index.d:
            print("mismatch:", query_arr.shape[1], self.index.d)
            # query_arr = np.resize(query_arr, (query_arr.shape[0], self.index.d))
        _, indices = self.index.search(query_arr, len(self.chunks))
        ranks = [[self.chunks[i] for i in indices[j]] for j in range(len(answers))]
        # calculate mean similarity for each chunk per answer
        ranks = [
            [(np.mean([np.inner(query_vectors[j], self.model.encode([chunk])[0]) for chunk in rank]), i) for i, rank in
             enumerate(ranks) for j in range(len(query_vectors))]]
        # sort by mean similarity
        ranks = [sorted(rank, key=lambda x: x[0]) for rank in ranks]
        # get the sorted indices
        ranks = [[index for _, index in rank] for rank in ranks]
        # get the top k chunks7
        if self.top_k > len(self.chunks):
            return self.chunks

        return [self.chunks[i] for i in ranks[0][-self.top_k:] if i in range(len(self.chunks))], answers

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
            print("Too many requests. Waiting 60 seconds.")
            time.sleep(60)
            return self.get_guesses(query)

        response_str = response.json()[0]["generated_text"].split("[/INST]")[-1].strip()

        r_list = response_str.split(";\n")

        if len(r_list) == 1:
            r_list = r_list[0].split("\n")

        if len(r_list) < self.num_of_paraphrases:
            r_list += ["[MASK]"] * (self.num_of_paraphrases - len(r_list))

        return r_list
