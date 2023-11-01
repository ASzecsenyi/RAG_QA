from data.NewsQaDocument import NewsQaDocument
from experiments.Experiment import Experiment
from qa.MistralQA import MistralQA
from qa.GptQA import GptQA
from retrieval.Chunker.CharChunker import CharChunker
from retrieval.Ranker.TfidfRanker import TfidfRanker

import json


document = NewsQaDocument(name="newsqa", story_id='./cnn/stories/289a45e715707cf650352f3eaa123f85d3653d4b.story')

chunker = CharChunker(chunk_length=100, sliding_window_size=0.5)
chunker2 = CharChunker(chunk_length=200, sliding_window_size=0.5)

ranker = TfidfRanker(top_k=5)

mi_qa = MistralQA(name="mistralqa")

gpt_qa = GptQA(name="gptqa")

experiment = Experiment(
    name="test",
    description="test",
    dataset=document,
    chunker=[chunker, chunker2],
    ranker=ranker,
    qa=[mi_qa, gpt_qa]
)

results = experiment.run()

# pretty print results
print(json.dumps(results, indent=4, sort_keys=True))
