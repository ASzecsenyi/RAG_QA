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

        story_questions = newsqa[split].filter(lambda x: x['story_id'] == story_id)

        assert len(story_questions) > 0, "story_id must be in the dataset"

        for question in story_questions:
            if len(question['story_text']) != len(story_questions['story_text'][0]):
                # convert answer_token_ranges to be relative to the story_text[0]
                # answer_token_ranges is a string of the form "start:end,start2:end2"
                # convert to a list of ints
                answer_token_ranges = question['answer_token_ranges'].split(',')
                answer_token_ranges = [answer_token_range.split(':') for answer_token_range in answer_token_ranges]
                # convert to ints
                answer_token_ranges = [
                    [int(answer_token_range[0]), int(answer_token_range[1])]
                    for answer_token_range in answer_token_ranges
                ]
                # get the answer span from the story_text
                for answer_token_range in answer_token_ranges:
                    answer_span = question['story_text'][answer_token_range[0]:answer_token_range[1]]
                    # find the answer span in the first story_text
                    answer_token_start = story_questions['story_text'][0].index(answer_span)
                    answer_token_end = answer_token_start + len(answer_span)
                    # update the answer_token_ranges
                    answer_token_range[0] = answer_token_start
                    answer_token_range[1] = answer_token_end
                # convert back to a string
                answer_token_ranges = [
                    f"{answer_token_range[0]}:{answer_token_range[1]}"
                    for answer_token_range in answer_token_ranges
                ]
                question['answer_token_ranges'] = ','.join(answer_token_ranges)

        document = story_questions['story_text'][0]

        questions: list[dict[str, Any]] = []
        for question in story_questions:
            questions.append({"question": question['question'], "answer_token_ranges": question['answer_token_ranges']})

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

