from django.shortcuts import render

from experiments import answer_single_question
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
            text_document = experiment.experimenttextdocument_set.first()
            chunker_params = experiment.experimentchunker_set.first()
            ranker_params = experiment.experimentranker_set.first()
            qa_params = experiment.experimentqa_set.first()

            if all([text_document, chunker_params, ranker_params, qa_params]):
                # text_document either has a file or a file_path
                if text_document.file and False:
                    file_data = TextDocument(text_document.file.read().decode('utf-8'))
                else:
                    with open(text_document.file_path, 'r') as file:
                        file_data = TextDocument(file.read())
                chunker = CharChunker(chunk_length=chunker_params.chunk_length,
                                      sliding_window_size=chunker_params.sliding_window_size)
                ranker = TfidfRanker(ranker_params.top_k)
                qa = MistralQA(qa_params.model_name)

                answer = answer_single_question(text_document.question, file_data, chunker, ranker, qa)

    return render(request, 'experiment_app/experiment_create_view.html', {
        'form': ExperimentForm(),
        'answer': answer,
    })
