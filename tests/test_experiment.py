import pytest
from unittest import mock
from qa import QA
from retrieval.Chunker import Chunker
from retrieval.Ranker import Ranker
from data import Document
from experiments.Experiment import Experiment, evaluate_rouge_score


class TestChunker(Chunker):
    def __init__(self):
        super().__init__(10)

    def chunk(self, document: str) -> list[str]:
        return [document[i:i + self.chunk_length] for i in range(0, len(document), self.chunk_length)]


class TestRanker(Ranker):
    def __init__(self):
        super().__init__(10)

    def init_chunks(self, chunks: list[str]):
        self.chunks = chunks

    def rank(self, query: str) -> list[str]:
        return self.chunks[:self.top_k]


class TestQA(QA):
    def predict(self, question: str, chunks: list[str]) -> str:
        return 'test answer'


def test_experiment_run():
    document = Document('test document', 'test1', [{'question': 'Who?', 'ground_truths': 'Test.'}])
    chunker = [TestChunker()]
    ranker = [TestRanker()]
    qa = [TestQA()]

    experiment = Experiment('test_experiment', 'test_description', document, chunker, ranker, qa)
    r = experiment.run()

    assert 'test_experiment' in experiment.name
    assert 'test_description' in experiment.description
    assert r == {
        'TestChunker_10_0.0_TestRanker_10_TestQA_test1': [
            {
                'answer': 'test answer',
                'contexts': ['test docum', 'ent'],
                'ground_truths': 'Test.',
                'question': 'Who?'
            }
        ],
        'times': {
            'Chunking': [0.0],
            'Initialising ranker': [0.0],
            'Qa': [0.0],
            'Ranking': [0.0]
        }
    }


def test_experiment_evaluate():
    experiment = Experiment('test_experiment', 'test_description', [], [], [], [])
    experiment.results = {
        'TestChunker_10_0.0_TestRanker_10_TestQA_test1': [
            {'question': 'Who?', 'answer': 'test answer', 'ground_truths': ['Test.'], 'contexts': ['test docum']}
        ]
    }

    # Mock the evaluate function from the ragas module.
    with mock.patch(
            'experiments.Experiment.evaluate_ragas_score',
    ) as mock_evaluate:
        mock_evaluate.return_value = {'em': 1, 'f1': 0.5, 'prec': 0.5, 'recall': 0.5}
        evaluations = experiment.evaluate_with_ragas()
        assert evaluations == {
            'TestChunker_10_0.0_TestRanker_10_TestQA_test1': {'em': 1, 'f1': 0.5, 'prec': 0.5, 'recall': 0.5}
        }


def test_evaluate_rouge_score():
    answers = ['test answer']
    ground_truths = ['Test.']
    scores = evaluate_rouge_score(answers, ground_truths)

    assert isinstance(scores, dict)
    assert 'fmeasure' in scores
    assert 'precision' in scores
    assert 'recall' in scores
