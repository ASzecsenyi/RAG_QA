import time

from retrieval.chunking.CharChunker import CharChunker
from retrieval.ranking.TfidfRanker import TfidfRanker

start_time = time.time()

chunker = CharChunker(chunk_length=20, sliding_window_size=0.5)

ranker = TfidfRanker(top_k=5)

document = 'This is a test document. It is used to test the retrieval system. It is a very short document.'

ranker.init_chunks(chunker.chunk(document))

query = 'test'

print(ranker.rank(query))

print("--- %s seconds ---" % (time.time() - start_time))
