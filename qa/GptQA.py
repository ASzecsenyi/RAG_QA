import time
from typing import Union

from qa import QA

import os
import openai


class GptQA(QA):
    """
    Gpt QA model.
    """
    default_prompt = (
        "You are a question answering agent."
        "Answer the question based on the context. "
        "Be as concise as possible, I cannot read too long answers. "
        "If you cannot answer the question, say [UNKNOWN]. "
        "If the question can be answered with yes or no, only say Yes or No."
        "Otherwise answer with a single sentence."
        "Only show your final answer! Do not provide any explanation or reasoning."
        "<SPLIT>"
        "CONTEXT: {chunks} "
        ""
        "QUESTION: {question} "
        ""
        "ANSWER: ")

    def __init__(self, name: str, prompt: str = default_prompt, api_key: str = None):
        super().__init__(name)
        self.api_key = api_key
        self.prompt = prompt
        if api_key is None:
            try:
                self.api_key = os.environ["OPENAI_API_KEY"]
            except KeyError:
                raise KeyError(
                    "OPENAI_API_KEY environment variable not found. Please set it or pass it as an argument."
                )
        if name is None:
            self.name += f"_{self.api_key[-5:]}"

        openai.api_key = self.api_key

    def predict(self, question: str, chunks: list[str], logprobs: bool = False) -> Union[str, tuple[str, list]]:
        """
        Predicts the answer to a question given a context.

        :param question: the question
        :type question: str
        :param chunks: the context
        :type chunks: list[str]
        :param logprobs: whether to return log probabilities
        :type logprobs: bool

        :return: the answer
        :rtype: str
        """

        system = self.prompt.split("<SPLIT>")[0]
        user = self.prompt.split("<SPLIT>")[1].format(chunks=" ".join(chunks), question=question)

        messages = [
            {"role": "system", "content": str(system)},
            {"role": "user", "content": str(user)},
        ]
        try:
            response = openai.chat.completions.create(
                model="gpt-3.5-turbo",
                temperature=0,
                max_tokens=100,
                stop=["\n"],
                messages=messages,
                logprobs=logprobs,
            )
        except Exception as e:
            print(e)
            print("Too many requests. Waiting 10 seconds.")
            time.sleep(10)
            return self.predict(question, chunks)

        if logprobs:
            return response.choices[0].message.content, response.choices[0].logprobs.content

        return response.choices[0].message.content
