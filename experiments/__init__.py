from typing import Union

from data import Document
from data.TextDocument import TextDocument
from qa import QA
from qa.MistralQA import MistralQA
from retrieval import Chunker, Ranker
from retrieval.Chunker.CharChunker import CharChunker
from retrieval.Ranker.TfidfRanker import TfidfRanker


def answer_single_question(
        question: str,
        dataset: Union[Document, str],
        chunker: Chunker = CharChunker(chunk_length=100, sliding_window_size=0.0),
        ranker: Ranker = TfidfRanker(top_k=5),
        qa: QA = MistralQA("default"),

):
    """
    Answers a single question with a given chunker, ranker and qa.

    :param question: The question
    :type question: str
    :param dataset: The dataset
    :type dataset: Union[Document, str]

    :param chunker: The chunker to use, defaults to CharChunker(chunk_length=100, sliding_window_size=0.0)
    :type chunker: Chunker, optional
    :param ranker: The ranker to use, defaults to TfidfRanker(5)
    :type ranker: Ranker, optional
    :param qa: The qa model to use, defaults to MistralQA("default")
    :type qa: QA, optional

    :return: The answer
    """

    # print(type(dataset))
    # print(dataset[:5])

    if isinstance(dataset, str):
        dataset = TextDocument(dataset)

    # get chunks
    chunks = chunker.chunk(document=dataset.document)

    # print(type(chunks))
    # print(chunks[:5])

    # init ranker
    ranker.init_chunks(chunks=chunks)

    # get ranked chunks
    chunks = ranker.rank(query=question)

    # predict answer
    answer = qa.predict(question=question, chunks=chunks)

    return answer
