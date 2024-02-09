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
    answer = None  # Use None as the default state to signify no answer yet

    if request.method == 'POST':
        form = ExperimentForm(request.POST, request.FILES)
        if form.is_valid():
            experiment = form.save()

            datasets = []

            for text_document in experiment.experimenttextdocument_set.all():
                file_data = TextDocument(
                    document=text_document.file_content,
                    questions=[{"question": text_document.question, "ground_truths": [""]}]
                )
                datasets.append(file_data)

            for news_qa_document in experiment.experimentnewsqadocument_set.all():
                file_data = NewsQaDocument(news_qa_document.story_id)
                datasets.append(file_data)

            chunkers = []
            for chunker_params in experiment.experimentchunker_set.all():
                chunkers.append(evals[chunker_params.chunker_type](
                    chunk_length=chunker_params.chunk_length,
                    sliding_window_size=chunker_params.sliding_window_size
                ))

            rankers = []
            for ranker_params in experiment.experimentranker_set.all():
                rankers.append(evals[ranker_params.ranker_type](ranker_params.top_k))

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

            results = experiment_run.run()

            # print(results)
            first_result = results[list(results.keys())[0]]
            if len(first_result) > 0:
                answer = first_result[0].get('answer', 'no ans')
            else:
                answer = 'no ans 2'

    form = ExperimentForm()

    grouped_fields = form.group_component_fields()

    initial_data = form.get_initial_components_data()

    return render(request, 'experiment_app/experiment_create_view.html', {
        'form': form,
        'answer': answer,
        'grouped_fields': grouped_fields,
        'initial_data': initial_data
    })
