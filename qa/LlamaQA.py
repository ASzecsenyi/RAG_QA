import os
import time

import requests

from qa import QA


class LlamaQA(QA):
    """
    Llama QA model.
    """

    default_prompt = ("<s>[INST] {chunks} [/INST] "
                      "Thank you, I will now answer your question based on this information with a single short sentence. "
                      "</s><s>"
                      "[INST] {question} [/INST]")

    def __init__(self, name: str, prompt: str = default_prompt, api_key: str = None):
        super().__init__(name)
        self.api_key = api_key
        self.prompt = prompt
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

        inputs = self.prompt.format(chunks=" ".join(chunks), question=question)

        time.sleep(0.1)
        try:
            response = requests.post(
                "https://api-inference.huggingface.co/models/meta-llama/Llama-2-7b-chat-hf",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={"inputs": inputs,
                      "parameters": {"max_new_tokens": 100}})
        except requests.exceptions.ConnectionError:
            print("Paused due to lost connection. Press enter to continue.")
            return self.predict(question, chunks)

        try:
            response.raise_for_status()
        except requests.exceptions.HTTPError as e:
            print(e)
            # if too many requests are made, the server will return a 429 status code
            print("Too many requests. Waiting 10 seconds.")
            time.sleep(10)
            return self.predict(question, chunks)

        return response.json()[0]["generated_text"].split("[/INST]")[-1].strip()

