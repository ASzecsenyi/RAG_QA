import os
import time

import requests

from qa import QA


class GemmaQA(QA):
    """
    Mistral QA model.
    """

    default_prompt = (
        "You are a question answering agent. "
        "Generate your response by following the steps below: "
        "1. Read the context and the question. "
        "2. Select the most relevant information from the context. "
        "3. Determine whether the question can be answered based on the context. "
        "4. If the question can be answered, generate a draft answer. "
        "5. Validate that the draft answer is completely grounded in the context. "
        "6. Generate your final answer and include the exact text from the context that supports your answer. Generate a single-sentence response that is clear, concise, and helpful. "
        "7. If the question cannot be answered with the context, say [UNKNOWN]. This will be your final answer in this case. "
        "8. Only show your final answer! Do not provide any explanation or reasoning. "
        ""
        "CONTEXT: {chunks} "
        ""
        "QUESTION: {question} "
        ""
        "FINAL_ANSWER: ")

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
                "https://api-inference.huggingface.co/models/google/gemma-7b-it",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={
                    "inputs": inputs,
                    "parameters": {
                        "max_new_tokens": 100,
                    }
                }
            )
        except requests.exceptions.ConnectionError:
            print("Paused due to lost connection. Press enter to continue.")
            return self.predict(question, chunks)

        try:
            response.raise_for_status()
        except requests.exceptions.HTTPError:
            # if too many requests are made, the server will return a 503 status code
            print("Too many requests. Waiting 10 seconds.")
            time.sleep(10)
            return self.predict(question, chunks)

        return response.json()[0]["generated_text"].split("[/INST]")[-1].strip().split("FINAL_ANSWER: ")[-1].strip()
