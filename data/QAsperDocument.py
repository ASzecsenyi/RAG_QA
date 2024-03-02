from typing import Any, Literal

from data import Document
from datasets import load_dataset

qasper = load_dataset("allenai/qasper")

qasper_top_200 = [
    "1911.10742",
    "1904.09131",
    "1611.06322",
    "1604.02038",
    "1911.04474",
    "1905.00840",
    "1810.02229",
    "1909.00091",
    "1909.04387",
    "2003.13016",
    "1805.11937",
    "1909.09070",
    "1708.05521",
    "1908.11049",
    "1907.10676",
    "1906.08871",
    "2004.04124",
    "1603.07252",
    "1708.00549",
    "1905.00472",
    "1912.02866",
    "1812.00382",
    "1903.02930",
    "1911.04873",
    "1606.07043",
    "1611.04234",
    "1909.00437",
    "2003.07568",
    "1810.02268",
    "1909.12079",
    "2003.11687",
    "1703.10152",
    "1907.04072",
    "1909.10481",
    "1805.04833",
    "1805.07882",
    "2004.01820",
    "1806.04387",
    "1808.04122",
    "1907.05338",
    "2003.08437",
    "2003.04978",
    "1809.08935",
    "1910.07924",
    "1911.11899",
    "1603.09405",
    "1912.11585",
    "1707.09816",
    "1703.04009",
    "1911.03090",
    "1909.04242",
    "2003.12139",
    "1809.03391",
    "1906.04287",
    "1912.10162",
    "1909.13184",
    "1910.13793",
    "1906.11085",
    "1908.09892",
    "2004.01970",
    "1806.03369",
    "2003.07459",
    "1803.08614",
    "1909.08250",
    "1612.07486",
    "1612.03762",
    "1906.01010",
    "2003.11528",
    "1801.03615",
    "1910.09362",
    "1911.12559",
    "1802.09059",
    "1803.09000",
    "1909.06708",
    "1705.10754",
    "1706.08568",
    "1909.05190",
    "1606.02601",
    "1602.04341",
    "1908.02284",
    "1907.00168",
    "1911.09709",
    "1909.11232",
    "1808.09180",
    "1909.04251",
    "1912.11980",
    "2004.04060",
    "1807.08089",
    "1703.05320",
    "1910.10762",
    "1711.01567",
    "1706.04206",
    "1707.06875",
    "1708.07690",
    "1804.05253",
    "1805.11598",
    "1610.03955",
    "1610.03807",
    "1607.03895",
    "1704.06960",
    "1904.03670",
    "2004.02363",
    "1601.03313",
    "1910.01108",
    "1707.06519",
    "1710.07394",
    "1912.06905",
    "1708.03312",
    "1908.05731",
    "1808.10059",
    "1911.12848",
    "1911.01770",
    "1903.10548",
    "1908.06893",
    "2002.12699",
    "1901.01590",
    "1603.03876",
    "1708.01065",
    "1905.07562",
    "1911.12893",
    "1910.07973",
    "1603.08868",
    "1910.01340",
    "1709.09749",
    "2003.09244",
    "1709.04491",
    "1908.07491",
    "1805.11850",
    "1710.06923",
    "1811.04791",
    "1906.01749",
    "2004.02334",
    "1908.11046",
    "1906.01076",
    "1908.09919",
    "1910.10670",
    "1902.06734",
    "1907.08540",
    "1611.04887",
    "1808.10006",
    "1702.02584",
    "1709.05453",
    "2002.06854",
    "1906.05506",
    "1808.00957",
    "1808.07231",
    "1712.02555",
    "1909.02322",
    "1805.04579",
    "1908.10149",
    "1803.07828",
    "1803.08419",
    "1908.08917",
    "1804.07445",
    "1910.03355",
    "1603.09381",
    "1708.05482",
    "1707.05589",
    "1910.14589",
    "1801.05617",
    "1905.08067",
    "1910.03943",
    "1911.12722",
    "1909.00786",
    "1909.03087",
    "2004.02105",
    "1909.01720",
    "1908.10090",
    "1905.05644",
    "1910.07154",
    "1901.09501",
    "1705.02023",
    "1912.06203",
    "1809.05807",
    "1904.02954",
    "1804.03839",
    "2004.02214",
    "1911.05652",
    "1703.10090",
    "1905.10039",
    "1704.06851",
    "1911.06815",
    "1704.08390",
    "1911.03642",
    "1911.01214",
    "1701.03578",
    "1802.09233",
    "1902.08830",
    "1909.08357",
    "1801.04433",
    "1908.07888",
    "2003.03131",
    "1903.01411",
    "2004.03090",
    "1709.06365",
    "1901.10619",
    "1612.09535",
    "1910.02001",
    "1603.01547",
    "1912.00239",
]


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

        self.paragraphs = {a: ' '.join(b) for a, b in zip(story['full_text']['section_name'], story['full_text']['paragraphs'])}

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
