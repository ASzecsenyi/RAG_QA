from django.shortcuts import render

from experiments.Experiment import Experiment

from data.TextDocument import TextDocument
from data.NewsQaDocument import NewsQaDocument, newsqa_top_300
from data.QAsperDocument import QAsperDocument, qasper_top_200

from retrieval.Chunker.CharChunker import CharChunker
from retrieval.Chunker.WordChunker import WordChunker
from retrieval.Chunker.SentChunker import SentChunker

from retrieval.Ranker.TfidfRanker import TfidfRanker
from retrieval.Ranker.CrossEncodingRanker import CrossEncodingRanker
from retrieval.Ranker.SentEmbeddingRanker import SentEmbeddingRanker
from retrieval.Ranker.GuessSimilarityRanker import GuessSimilarityRanker
from retrieval.Ranker.HybridRanker import HybridRanker
from retrieval.Ranker.PromptRanker import PromptRanker

from qa.MistralQA import MistralQA
from qa.LlamaQA import LlamaQA
from qa.GptQA import GptQA
from qa.GemmaQA import GemmaQA


from .forms import ExperimentForm

evals = {
    'CharChunker': CharChunker,
    'WordChunker': WordChunker,
    'SentChunker': SentChunker,
    'TfidfRanker': TfidfRanker,
    'SentEmbeddingRanker': SentEmbeddingRanker,
    'CrossEncodingRanker': CrossEncodingRanker,
    'GuessSimilarityRanker': GuessSimilarityRanker,
    'HybridRanker': HybridRanker,
    'PromptRanker': PromptRanker,
    'MistralQA': MistralQA,
    'GptQA': GptQA,
    'GemmaQA': GemmaQA,
    'LlamaQA': LlamaQA
}

doc_evals = {
    'NewsQaDocument': NewsQaDocument,
    'QAsperDocument': QAsperDocument,
}

# from .models import Experiment


def experiment_create_view(request):
    answers = {}

    if request.method == 'POST':
        form = ExperimentForm(request.POST, request.FILES)
        if form.is_valid():
            experiment = form.save()

            datasets = []

            for text_document in experiment.experimenttextdocument_set.all():
                file_data = TextDocument(
                    document=text_document.file_content,
                    questions=[{"question": text_document.question, "ground_truths": [""]}],
                    name=text_document.textdoc_name
                )
                datasets.append(file_data)

            for news_qa_document in experiment.experimentnewsqadocument_set.all():
                for story_id in range(news_qa_document.num_of_stories):
                    if news_qa_document.newsqa_type == 'NewsQaDocument':
                        file_data = doc_evals[news_qa_document.newsqa_type](
                            story_id=newsqa_top_300[story_id],
                            split='train',
                            name=f"{news_qa_document.newsqa_name}_{story_id}"
                        )
                    else:
                        file_data = doc_evals[news_qa_document.newsqa_type](
                            story_id=qasper_top_200[story_id],
                            name=f"{news_qa_document.newsqa_name}_{story_id}"
                        )

                    datasets.append(file_data)

            chunkers = []
            for chunker_params in experiment.experimentchunker_set.all():
                chunkers.append(evals[chunker_params.chunker_type](
                    chunk_length=chunker_params.chunk_length,
                    sliding_window_size=chunker_params.sliding_window_size,
                    name=chunker_params.chunker_name
                ))

            rankers = []
            for ranker_params in experiment.experimentranker_set.all():
                rankers.append(evals[ranker_params.ranker_type](
                    top_k=ranker_params.top_k,
                    name=ranker_params.ranker_name
                ))

            qas = []
            for qa_params in experiment.experimentqa_set.all():
                qas.append(evals[qa_params.model_type](qa_params.model_name))

            experiment_run = Experiment(
                name=experiment.name,
                description=experiment.description,
                dataset=datasets,
                chunker=chunkers,
                ranker=rankers,
                qa=qas,
                # autoload=False
            )

            # print(experiment_run)

            experiment_run.run()
            experiment_run.evaluate_with_rouge_score()
            results = experiment_run.results



            for dataset_name in [text_document.textdoc_name for text_document in experiment.experimenttextdocument_set.all()]:
                for chunker_name in [chunker.chunker_name for chunker in experiment.experimentchunker_set.all()]:
                    for ranker_name in [ranker.ranker_name for ranker in experiment.experimentranker_set.all()]:
                        for qa_name in [qa.model_name for qa in experiment.experimentqa_set.all()]:
                            answers[f"{chunker_name}<sep>{ranker_name}<sep>{qa_name}<sep>{dataset_name}"] = results[f"{chunker_name}_{ranker_name}_{qa_name}_{dataset_name}"]
            for dataset in [news_qa_document for news_qa_document in experiment.experimentnewsqadocument_set.all()]:
                for story_id in range(dataset.num_of_stories):
                    for chunker_name in [chunker.chunker_name for chunker in experiment.experimentchunker_set.all()]:
                        for ranker_name in [ranker.ranker_name for ranker in experiment.experimentranker_set.all()]:
                            for qa_name in [qa.model_name for qa in experiment.experimentqa_set.all()]:
                                answers[f"{chunker_name}<sep>{ranker_name}<sep>{qa_name}<sep>{dataset.newsqa_name}_{story_id}"] = results[f"{chunker_name}_{ranker_name}_{qa_name}_{dataset.newsqa_name}_{story_id}"]

            print(results)

    form = ExperimentForm()

    initial_data = form.get_initial_components_data()

    return render(request, 'experiment_app/experiment_create_view.html', {
        'form': form,
        'answers': answers,
        'initial_data': initial_data
    })
