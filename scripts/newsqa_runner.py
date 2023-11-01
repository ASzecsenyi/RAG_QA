from datasets import load_dataset

import time

from retrieval.chunking.CharChunker import CharChunker
from retrieval.ranking.TfidfRanker import TfidfRanker

from qa.GptQA import GptQA
from qa.MistralQA import MistralQA

newsqa = load_dataset('newsqa', data_dir='../data/files')

document = newsqa['train'][0]['story_text']

query = newsqa['train'][0]['question']

start_time = time.time()

chunker = CharChunker(chunk_length=100, sliding_window_size=0.5)

ranker = TfidfRanker(top_k=5)

ranker.init_chunks(chunker.chunk(document))

chunks = ranker.rank(query)

print(chunks)

print(len(chunks))

print("--- %s seconds ---" % (time.time() - start_time))

start_time = time.time()

gpt_qa = GptQA(name="gptqa")
mi_qa = MistralQA(name="mistralqa")

print('gpt', gpt_qa.predict(query, chunks))
print('mistral', mi_qa.predict(query, chunks))

print("--- %s seconds ---" % (time.time() - start_time))


