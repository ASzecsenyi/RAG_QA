from qa import QA

import os
import openai


class GptQA(QA):
    """
    Gpt QA model.
    """

    def __init__(self, name: str, api_key: str = None):
        super().__init__(name)
        self.api_key = api_key
        if api_key is None:
            try:
                self.api_key = os.environ["OPENAI_API_KEY"]
            except KeyError:
                raise KeyError(
                    "OPENAI_API_KEY environment variable not found. Please set it or pass it as an argument."
                )

        openai.api_key = self.api_key

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

        system = f"{chunks}"
        user = f"{question}"

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
        )

        return response.choices[0].message["content"]



