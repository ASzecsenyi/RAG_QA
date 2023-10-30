import os

import requests

from qa import QA


class MistralQA(QA):
    """
    Mistral QA model.
    """

    def __init__(self, name: str, api_key: str = None):
        super().__init__(name)
        self.api_key = api_key
        if api_key is None:
            try:
                self.api_key = os.environ["HUGGINGFACE_API_KEY"]
            except KeyError:
                raise KeyError(
                    "HUGGINGFACE_API_KEY environment variable not found. Please set it or pass it as an argument."
                )

    def predict(self, question: str, chunks: list[str]) -> str:
        """
        Predicts the answer to a question given a context.

        :param question: the question
        :type question: str
        :param chunks: the context
        :type chunks: list[str]

        :return: the answer
        :rtype: str
        """

        inputs = (f"<s>[INST] {chunks} [/INST] "
                  f"Thank you, I will now answer your question based on this information. "
                  f"[INST] {question} [/INST]")

        response = requests.post(
            "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.1",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={"inputs": inputs})

        return response.json()[0]["generated_text"]