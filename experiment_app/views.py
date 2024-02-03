from django.shortcuts import render, redirect

from experiments import answer_single_question
from qa.MistralQA import MistralQA
from retrieval.Chunker.CharChunker import CharChunker
from retrieval.Ranker.TfidfRanker import TfidfRanker
# Create your views here.
from .forms import TextFileUploadForm, ChunkerForm, RankerForm, QAForm


def model_form_upload(request):
    if request.method == 'POST':
        form = TextFileUploadForm(request.POST, request.FILES)
        chunker_form = ChunkerForm(request.POST)
        ranker_form = RankerForm(request.POST)
        qa_form = QAForm(request.POST)
        answer = False
        if form.is_valid() and chunker_form.is_valid() and ranker_form.is_valid() and qa_form.is_valid():
            if form.cleaned_data['file']:
                uploaded_file = form.save()
                file_data = ""

                for chunk in uploaded_file.file.chunks():
                    file_data += chunk.decode('utf-8')

                # set the file path to the uploaded file
                uploaded_file.file_path = uploaded_file.file.path
            elif form.cleaned_data['file_path']:
                with open(form.cleaned_data['file_path'], 'r') as file:
                    file_data = file.read()

                uploaded_file = form.save()

            chunker = CharChunker(chunk_length=chunker_form.cleaned_data['chunk_length'],
                                  sliding_window_size=chunker_form.cleaned_data['sliding_window_size'])
            ranker = TfidfRanker(ranker_form.cleaned_data['num_rank'])
            qa = MistralQA(qa_form.cleaned_data['model_name'])

            answer = answer_single_question(question=form.cleaned_data['question'],
                                            dataset=file_data,
                                            chunker=chunker,
                                            ranker=ranker,
                                            qa=qa)
            # Save the form data, the answer, and the path to the uploaded file in the session
            request.session['form_data'] = request.POST
            request.session['answer'] = answer
            request.session['file_path'] = uploaded_file.file_path
            return redirect('experiment_app:model_form_upload')
    else:
        # Retrieve the form data, the answer, and the path to the uploaded file from the session
        form_data = request.session.get('form_data', {})
        answer = request.session.get('answer', False)
        file_path = request.session.get('file_path', None)
        form_data['file_path'] = file_path
        form = TextFileUploadForm(form_data)
        chunker_form = ChunkerForm(form_data)
        ranker_form = RankerForm(form_data)
        qa_form = QAForm(form_data)
    return render(request, 'experiment_app/model_form_upload.html', {
        'form': form,
        'chunker_form': chunker_form,
        'ranker_form': ranker_form,
        'qa_form': qa_form,
        'answer': answer
    })
