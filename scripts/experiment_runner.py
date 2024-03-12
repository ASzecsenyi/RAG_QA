from data.UXDocument import UXDocument, chapters
from data.TextDocument import TextDocument
from data.NewsQaDocument import NewsQaDocument, newsqa_top_300
from data.QAsperDocument import QAsperDocument, qasper_top_200
from experiments.Experiment import Experiment
from qa.MistralQA import MistralQA
from qa.GptQA import GptQA
from qa.GemmaQA import GemmaQA
from qa.LlamaQA import LlamaQA
from retrieval.Chunker.CharChunker import CharChunker
from retrieval.Chunker.SentChunker import SentChunker
from retrieval.Chunker.WordChunker import WordChunker
from retrieval.Ranker.TfidfRanker import TfidfRanker
from retrieval.Ranker.SentEmbeddingRanker import SentEmbeddingRanker
from retrieval.Ranker.GuessSimilarityRanker import GuessSimilarityRanker
from retrieval.Ranker.HybridRanker import HybridRanker
from retrieval.Ranker.CrossEncodingRanker import CrossEncodingRanker
from retrieval.Ranker.PromptRanker import PromptRanker
from retrieval.Ranker import RandomRanker

# documents = [UXDocument(name=f"ux_{chapter}", chapter=chapter) for chapter in chapters]
# document = UXDocument()


newsqa_documents = [NewsQaDocument(name=f"newsqa_{i}", story_id=story, split='train') for
             i, story in enumerate(newsqa_top_300[:100])]

qasper_documents = [QAsperDocument(name=f"qasper_{i}", story_id=qa) for i, qa in enumerate(qasper_top_200[:200])]

documents = newsqa_documents + qasper_documents

num_of_total_questions = sum([len(doc.questions) for doc in documents])
print(f"Total number of questions: {num_of_total_questions}")
num_of_newsqa_questions = sum([len(doc.questions) for doc in newsqa_documents])
print(f"Total number of NewsQA questions: {num_of_newsqa_questions}")
num_of_qasper_questions = sum([len(doc.questions) for doc in qasper_documents])
print(f"Total number of QAsper questions: {num_of_qasper_questions}")
qasper_answer_types = {}
for doc in qasper_documents:
    for question in doc.questions:
        answer_type = question['answer_type']
        if answer_type in qasper_answer_types:
            qasper_answer_types[answer_type] += 1
        else:
            qasper_answer_types[answer_type] = 1

print(qasper_answer_types)
chunker = SentChunker(chunk_length=2, sliding_window_size=0.5)
chunker1 = SentChunker(chunk_length=3, sliding_window_size=0.5)
chunker2 = SentChunker(chunk_length=4, sliding_window_size=0.5)

all_chunkers = [chunker, chunker1, chunker2]


tfidf_ranker = TfidfRanker(top_k=5)
# se_ranker = SentEmbeddingRanker(top_k=5)
ce_ranker = CrossEncodingRanker(top_k=5)
gs_ranker = GuessSimilarityRanker(top_k=5)
hy_ranker = HybridRanker(top_k=5)
random_ranker = RandomRanker(top_k=5)
p_ranker = PromptRanker(top_k=5)

all_rankers = [tfidf_ranker, ce_ranker, gs_ranker, hy_ranker, random_ranker, p_ranker]

mi_qa = MistralQA(name="mistralqa")
lam_qa = LlamaQA(name="llamaqa")
gem_qa = GemmaQA(name="gemmaqa")
gpt_qa = GptQA(name="gptqa", api_key="")

all_qas = [mi_qa, lam_qa, gem_qa, gpt_qa]

experiment = Experiment(
    name="Final_Overall_Experiment",
    description="Uniform prompts, All default rankers, SentChunker, 5 top_k, 100 newsqa, 200 qasper, 300 total"
                "06/03/2024: 21:52",
    dataset=documents,
    chunker=all_chunkers,
    ranker=all_rankers,
    qa=all_qas,
    # autoload=False
)

experiment.verbose = True

results = experiment.run(get_ground_ranks=True)
# results = experiment.load_results("../data/files/test.json")

# pretty print results
# print(json.dumps(results, indent=4, sort_keys=True))

# save results
# experiment.save_results()

evaluation = experiment.aggregate_evaluations()


experiment.save_results()

# print(json.dumps(evaluation, indent=4, sort_keys=True))
