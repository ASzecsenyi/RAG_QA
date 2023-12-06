import os

import requests

from qa import QA


class LlamaQA(QA):
    """
    Llama QA model.
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
                  f"Thank you, I will now very briefly answer your question based on this information. </s><s>"
                  f"[INST] {question} [/INST]")

        try:
            response = requests.post(
                "https://api-inference.huggingface.co/models/meta-llama/Llama-2-7b-chat-hf",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={"inputs": inputs,
                      "parameters": {"max_new_tokens": 100}})
        except requests.exceptions.ConnectionError:
            input("Paused due to lost connection. Press enter to continue.")
            return self.predict(question, chunks)

        response.raise_for_status()

        return response.json()[0]["generated_text"].split("[/INST]")[-1].strip()