import os
import time

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
        if name is None:
            self.name += f"_{self.api_key[-5:]}"

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
                  f"Thank you, I will now very briefly answer your question based on this information. "
                  f"[INST] {question} [/INST]")
        time.sleep(1)
        try:
            response = requests.post(
                "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.1",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={
                    "inputs": inputs,
                    "parameters": {
                        "max_new_tokens": 100,
                    }
                }
            )
        except requests.exceptions.ConnectionError:
            input("Paused due to lost connection. Press enter to continue.")
            return self.predict(question, chunks)

        response.raise_for_status()

        return response.json()[0]["generated_text"].split("[/INST]")[-1].strip()
