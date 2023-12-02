from data.NewsQaDocument import NewsQaDocument
from experiments.Experiment import Experiment
from qa.MistralQA import MistralQA
from qa.GptQA import GptQA
from retrieval.Chunker.CharChunker import CharChunker
from retrieval.Ranker.TfidfRanker import TfidfRanker

import json


document = NewsQaDocument(name="newsqa", story_id='./cnn/stories/289a45e715707cf650352f3eaa123f85d3653d4b.story')
document2 = NewsQaDocument(name="newsqa", story_id='./cnn/stories/bce33bb5b5cff6b93065aa0cf91917c8dd36ac78.story')
document3 = NewsQaDocument(name="newsqa", story_id='./cnn/stories/017df5c4fe1e79eb26957ff6a8b4c1e41cd966ac.story')

all_documents = NewsQaDocument.all_documents()

chunker = CharChunker(chunk_length=100, sliding_window_size=0.5)
chunker2 = CharChunker(chunk_length=200, sliding_window_size=0.5)

ranker = TfidfRanker(top_k=5)

mi_qa = MistralQA(name="mistralqa")

# gpt_qa = GptQA(name="gptqa")

experiment = Experiment(
    name="test",
    description="test",
    dataset=all_documents[:20],
    chunker=[chunker2],
    ranker=ranker,
    qa=[mi_qa]  # , gpt_qa]
)

experiment.verbose = True

results = experiment.load_results("../data/files/test_first20.json")

# pretty print results
print(json.dumps(results, indent=4, sort_keys=True))

# save results
experiment.save_results()

# evaluation = experiment.evaluate_with_ragas()

evaluation = experiment.evaluate_with_rouge_score()

print(json.dumps(evaluation, indent=4, sort_keys=True))
