import os
import time
import warnings

import numpy as np
import requests

from retrieval.Ranker import Ranker
from retrieval.Ranker.CrossEncodingRanker import CrossEncodingRanker


class PromptRanker(Ranker):
    default_prompt = ("<s>[INST] Question: {query}"
                      "Table of Contents:"
                      "{paragraphs}"
                      "Each heading (or line) in the table of contents above represents a fraction in a document."
                      "Select the five headings that help the best to find out the information for the question."
                      "List the headings in the order of importance and in the format of"
                      "'1. ---"
                      "2. ---"
                      "---"
                      "5. ---'."
                      "Don't say anything other than the format."
                      "If the question is about greetings or casual talks, just say 'Disregard the reference.'."
                      "[/INST] ")

    def __init__(
            self,
            top_k: int,
            ranker: Ranker = CrossEncodingRanker(top_k=5),
            prompt: str = default_prompt,
            api_key: str = None,
            name=None,
            **kwargs
    ):
        """
        :param top_k: The number of chunks to return
        :type top_k: int

        :param kwargs: Keyword arguments for the TfidfVectorizer
        :type kwargs: dict
        """
        super().__init__(top_k, name=name,)
        self.api_key = api_key
        self.prompt = prompt
        self.paragraphs = {}
        if api_key is None:
            try:
                self.api_key = os.environ["HUGGINGFACE_API_KEY"]
            except KeyError:
                raise KeyError(
                    "HUGGINGFACE_API_KEY environment variable not found. Please set it or pass it as an argument."
                )
        if name is None:
            self.name += f"_{ranker.name}" + f"_{kwargs}" if kwargs else ""

        self.model = ranker

    def init_chunks(self, chunks: list[str], paragraphs: dict[str, str] = None):
        self.chunks: list[str] = chunks
        self.paragraphs: dict[str, str] = paragraphs
        if paragraphs is None:
            self.paragraphs = {}
            warnings.warn(f"No paragraphs were given. The ranker will function as a regular {self.model.name}.")

    def rank(self, query: str, return_similarities: bool = False) -> tuple[list[str], list[str]] | tuple[list[tuple[str, float]], list[str]]:
        paragraphs_of_choice: list[str] = self.get_paragraphs(query)

        # print(self.paragraphs.keys())

        # print(paragraphs_of_choice)

        # get chunks that are in the paragraphs
        chunks = [chunk for chunk in self.chunks if any(chunk.replace(' ', '') in self.paragraphs[paragraph].replace(' ', '') for paragraph in paragraphs_of_choice)]

        # add chunks before and after the selected chunks to completely cover the paragraphs
        for i, chunk in enumerate(self.chunks):
            if chunk in chunks:
                if i > 0 and self.chunks[i-1] not in chunks:
                    chunks.append(self.chunks[i-1])
                if i < len(self.chunks) - 1 and self.chunks[i+1] not in chunks:
                    chunks.append(self.chunks[i+1])

        if len(chunks) == 0:
            # if the paragraphs are not in the chunks (i.e. they are smaller than their chunk), use the paragraphs
            chunks = [self.paragraphs[paragraph] for paragraph in paragraphs_of_choice]

        # all chunks if self.paragraphs is empty
        if self.paragraphs == {}:
            chunks = self.chunks
        print("")

        print(self.chunks)

        print(chunks)

        self.model.init_chunks(chunks)
        return self.model.rank(query, return_similarities=return_similarities), paragraphs_of_choice

    def get_paragraphs(self, query: str) -> list[str]:
        if self.paragraphs == {}:
            return []
        inputs = self.prompt.format(query=query, paragraphs=self.paragraphs.keys())

        try:
            response = requests.post(
                "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.1",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={"inputs": inputs,
                      "parameters": {"max_new_tokens": 100}})
        except requests.exceptions.ConnectionError:
            print("Paused due to lost connection. Press enter to continue.")
            return self.get_paragraphs(query)
        try:
            response.raise_for_status()
        except requests.exceptions.HTTPError:
            # if too many requests are made, the server will return a 503 status code
            print("Too many requests. Waiting 10 seconds.")
            time.sleep(10)
            return self.get_paragraphs(query)

        response_str = response.json()[0]["generated_text"].split("[/INST]")[-1].strip()

        r_list = response_str.split(";\n")

        if len(r_list) == 1:
            r_list = r_list[0].split("\n")
        if len(r_list) == 1:
            r_list = r_list[0].split(";")

        # remove leading ordinal numbers
        r_list = [x.split(". ", 1)[1] if x[1] == "." else x for x in r_list]

        return r_list

    def batch_rank(self, queries: list[str], batch_size: int = 100) -> list[list[str]]:
        raise NotImplementedError
