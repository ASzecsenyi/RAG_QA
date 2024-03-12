# Retrieval Strategies

This directory holds retrieval components used in the RAG_QA application.

In `Chunker/` can be found the different chunkers used to split the documents into smaller parts.

- `CharChunker.py`: Splits the document into chunks of a fixed number of characters.
- `WordChunker.py`: Splits the document into chunks of a fixed number of words.
- `SentChunker.py`: Splits the document into chunks of a fixed number of sentences.

These implement the `Chunker` abstract class found in `Chunker/__init__.py`.

In `Ranker/` can be found the different rankers used to order the chunks.

- `TfIdfRanker.py`: Ranks the chunks using the TF-IDF algorithm.
- `SentEmbedingRanker.py`: Ranks the chunks using sentence embeddings.
- `CrossEncodingRanker.py`: Ranks the chunks using a cross-encoder model.
- `GuessSimilarityRanker.py`: Ranks the chunks using a guess from an LLM about what the answer might look like.
- `HybridRanker.py`: Ranks the chunks using a hybrid model that combines sparse and dense embeddings.
- `PromptRanker.py`: Ranks the chunks after looking at the table of contents with an LLM.

These implement the `Ranker` abstract class found in `Ranker/__init__.py`.