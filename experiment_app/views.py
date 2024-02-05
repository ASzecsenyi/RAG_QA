from django.shortcuts import render

from experiments import answer_single_question
from experiments.Experiment import Experiment
from qa.MistralQA import MistralQA
from retrieval.Chunker.CharChunker import CharChunker
from retrieval.Ranker.TfidfRanker import TfidfRanker
from data.TextDocument import TextDocument

from .forms import ExperimentForm


# from .models import Experiment


def experiment_create_view(request):
    answer = None  # Use None as the default state to signify no answer yet

    if request.method == 'POST':
        form = ExperimentForm(request.POST, request.FILES)
        if form.is_valid():
            experiment = form.save()
            # text_document = experiment.experimenttextdocument_set.first()
            # chunker_params = experiment.experimentchunker_set.first()
            # ranker_params = experiment.experimentranker_set.first()
            # qa_params = experiment.experimentqa_set.first()
            #
            # if all([text_document, chunker_params, ranker_params, qa_params]):
            #     # text_document either has a file or a file_path
            #     if text_document.file and False:
            #         file_data = TextDocument(text_document.file.read().decode('utf-8'))
            #     else:
            #         with open(text_document.file_path, 'r') as file:
            #             file_data = TextDocument(file.read())
            #     chunker = CharChunker(chunk_length=chunker_params.chunk_length,
            #                           sliding_window_size=chunker_params.sliding_window_size)
            #     ranker = TfidfRanker(ranker_params.top_k)
            #     qa = MistralQA(qa_params.model_name)
            #
            #     answer = answer_single_question(text_document.question, file_data, chunker, ranker, qa)

            datasets = []

            for text_document in experiment.experimenttextdocument_set.all():
                file_data = TextDocument(
                    document=text_document.file_content,
                    questions=[{"question": text_document.question, "ground_truths": [""]}]
                )
                datasets.append(file_data)

            chunkers = []
            for chunker_params in experiment.experimentchunker_set.all():
                chunkers.append(CharChunker(chunk_length=chunker_params.chunk_length,
                                            sliding_window_size=chunker_params.sliding_window_size))

            rankers = []
            for ranker_params in experiment.experimentranker_set.all():
                rankers.append(TfidfRanker(ranker_params.top_k))

            qas = []
            for qa_params in experiment.experimentqa_set.all():
                qas.append(MistralQA(qa_params.model_name))

            experiment_run = Experiment(
                name=experiment.name,
                description=experiment.description,
                dataset=datasets,
                chunker=chunkers,
                ranker=rankers,
                qa=qas
            )

            print(experiment_run)

            results = experiment_run.run()

            # print(results)
            first_result = results[list(results.keys())[0]]
            if len(first_result) > 0:
                answer = first_result[0].get('answer', 'noans')
            else:
                answer = 'noansss'

    return render(request, 'experiment_app/experiment_create_view.html', {
        'form': ExperimentForm(),
        'answer': answer,
    })
