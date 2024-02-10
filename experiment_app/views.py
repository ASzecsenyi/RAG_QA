from django.shortcuts import render

from experiments.Experiment import Experiment
from data.TextDocument import TextDocument
from data.NewsQaDocument import NewsQaDocument
from retrieval.Chunker.CharChunker import CharChunker
from retrieval.Chunker.SentChunker import SentChunker
from retrieval.Ranker.TfidfRanker import TfidfRanker
from retrieval.Ranker.SentEmbeddingRanker import SentEmbeddingRanker
from retrieval.Ranker.GuessSimilarityRanker import GuessSimilarityRanker
from qa.MistralQA import MistralQA
from qa.LlamaQA import LlamaQA
from qa.GptQA import GptQA

from .forms import ExperimentForm

evals = {
    'CharChunker': CharChunker,
    'SentChunker': SentChunker,
    'TfidfRanker': TfidfRanker,
    'SentEmbeddingRanker': SentEmbeddingRanker,
    'GuessSimilarityRanker': GuessSimilarityRanker,
    'MistralQA': MistralQA,
    'GptQA': GptQA,
    'LlamaQA': LlamaQA
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
                file_data = NewsQaDocument(
                    story_id=news_qa_document.story_id,
                    name=news_qa_document.newsqa_name
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
                qa=qas
            )

            # print(experiment_run)

            experiment_run.run()
            experiment_run.evaluate_with_rouge_score()
            results = experiment_run.results

            for chunker_name in [chunker.chunker_name for chunker in experiment.experimentchunker_set.all()]:
                for ranker_name in [ranker.ranker_name for ranker in experiment.experimentranker_set.all()]:
                    for qa_name in [qa.model_name for qa in experiment.experimentqa_set.all()]:
                        for dataset_name in [text_document.textdoc_name for text_document in experiment.experimenttextdocument_set.all()] + [news_qa_document.newsqa_name for news_qa_document in experiment.experimentnewsqadocument_set.all()]:
                            answers[f"{chunker_name}<sep>{ranker_name}<sep>{qa_name}<sep>{dataset_name}"] = results[f"{chunker_name}_{ranker_name}_{qa_name}_{dataset_name}"]

            print(results)

    form = ExperimentForm()

    initial_data = form.get_initial_components_data()

    return render(request, 'experiment_app/experiment_create_view.html', {
        'form': form,
        'answers': answers,
        # 'context': context,
        # 'result': result,
        'initial_data': initial_data
    })
