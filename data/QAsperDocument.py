from typing import Any, Literal

from data import Document
from datasets import load_dataset

qasper = load_dataset("allenai/qasper")


class QAsperDocument(Document):
    """
    A document from the NewsQA dataset.
    """

    def __init__(self, name: str = None, story_id: str = None, split: Literal['train', 'validation', 'test'] = 'test'):
        """
        :param name: the dataset identifier - name used in paper it is released with, e.g. "hotpot_qa"
        :type name: str, optional
        :param story_id: the id of the story to get data from
        :type story_id: str
        """

        if story_id is None:
            story_id = qasper[split][0]['id']

        if name is None:
            name = f"newsqa_{story_id}"

        print(name)

        # get the story
        story = None

        for i, s in enumerate(qasper[split]):
            if s['id'] == story_id:
                story = s
                break

        # assert len(story) == 1, "story_id must be in the dataset"

        # story = story[0]

        # it is assumed that all story_text instances with the same story_id are the same

        document = '\n\n'.join(['\n'.join(section) for section in story['full_text']['paragraphs']])

        questions: list[dict[str, Any]] = []
        for i, question in enumerate(story['qas']['question']):
            # get texts at the answer token ranges
            answer = story['qas']['answers'][i]['answer']
            evidence = []
            ground_truths = []
            # dict_keys(['unanswerable', 'extractive_spans', 'yes_no', 'free_form_answer', 'evidence', 'highlighted_evidence'])
            # print(answer)
            for i, ans in enumerate(answer):
                if ans['unanswerable']:
                    ground_truths.append("[UNKNOWN]")
                elif len(ans['extractive_spans']) > 0:
                    ground_truths += ans['extractive_spans']
                elif ans['yes_no'] is not None:
                    ground_truths.append(str(ans['yes_no']))
                elif len(ans['free_form_answer']) > 0:
                    ground_truths.append(ans['free_form_answer'])
                evidence.append(ans['highlighted_evidence'])

            questions.append(
                {
                    "question": question,
                    "evidence": evidence,
                    "ground_truths": ground_truths,
                }
            )

        super().__init__(document, name, questions)
