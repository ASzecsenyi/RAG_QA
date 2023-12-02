from typing import Any

from tqdm import tqdm

from data import Document
from datasets import load_dataset

import pandas as pd

newsqa = load_dataset('newsqa', data_dir='../data/files')


class NewsQaDocument(Document):
    """
    A document from the NewsQA dataset.
    """

    def __init__(self, name: str = None, story_id: str = None, split: str = 'test'):
        """
        :param name: the dataset identifier - name used in paper it is released with, e.g. "hotpot_qa"
        :type name: str, optional
        :param story_id: the id of the story to get data from
        :type story_id: str
        """

        if story_id is None:
            story_id = newsqa[split][0]['story_id']

        if name is None:
            name = f"newsqa_{story_id}"

        # get all instances of the story and questions about it
        story_questions = newsqa[split].filter(lambda x: x['story_id'] == story_id)

        assert len(story_questions) > 0, "story_id must be in the dataset"

        # it is assumed that all story_text instances with the same story_id are the same

        document = story_questions['story_text'][0]

        questions: list[dict[str, Any]] = []
        for question in story_questions:
            # get texts at the answer token ranges
            answer_token_ranges = [
                [int(token_range.split(':')[0]), int(token_range.split(':')[1])]
                for token_range in question['answer_token_ranges'].split(',')]
            document_tokens = document.split(' ')
            ground_truths = [' '.join(document_tokens[start:end]) for start, end in answer_token_ranges]

            questions.append(
                {
                    "question": question['question'],
                    "answer_token_ranges": question['answer_token_ranges'],
                    "ground_truths": ground_truths,
                }
            )

        super().__init__(document, name, questions)

    @classmethod
    def all_documents(cls, split: str = 'test', max_stories: int = None) -> list["NewsQaDocument"]:
        """
        Creates a list of all unique documents in the dataset.

        :return: the list of documents
        :rtype: list[Document]
        :param split: the split to get documents from
        :type split: str
        :param max_stories: the maximum number of documents to get, defaults to None - gets all documents
        :type max_stories: int, optional
        """

        documents = []

        unique_story_ids = pd.unique(newsqa[split]['story_id'])

        if max_stories is not None:
            unique_story_ids = unique_story_ids[:max_stories]

        for story_id in tqdm(unique_story_ids):
            try:
                documents.append(cls(story_id=story_id, split=split))
            except ValueError as e:
                print(e)
        return documents

