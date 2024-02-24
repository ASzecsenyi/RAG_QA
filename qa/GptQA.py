from typing import Union

from qa import QA

import os
import openai


class GptQA(QA):
    """
    Gpt QA model.
    """
    default_prompt = ("You are a question answering agent."
                      "Generate your response by following the steps below: "
                      "1. Read the context and the question. "
                      "2. Select the most relevant information from the context. "
                      "3. Determine whether the question can be answered based on the context. "
                      "4. If the question can be answered, generate a draft answer. "
                      "5. Validate that the draft answer is completely grounded in the context. "
                      "6. Generate your final answer and include the exact text from the context that supports your answer. Generate a single-sentence response that is clear, concise, and helpful. "
                      "7. If the question cannot be answered with the context, say [UNKNOWN]. This will be your final answer in this case. "
                      "8. Only show your final answer! Do not provide any explanation or reasoning. "
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
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            temperature=0,
            max_tokens=100,
            stop=["\n"],
            messages=messages,
            logprobs=logprobs,
        )

        if logprobs:
            return response.choices[0].message["content"], response.choices[0].logprobs["content"]

        return response.choices[0].message["content"]



