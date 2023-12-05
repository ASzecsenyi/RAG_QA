from django.shortcuts import render

from experiments import answer_single_question
# Create your views here.
from .forms import TextFileUploadForm


def model_form_upload(request):
    if request.method == 'POST':
        form = TextFileUploadForm(request.POST, request.FILES)
        if form.is_valid():
            form.save()
            # process the file here and return the answer
            # use the function answer_single_question to generate the answer
            # and pass it to template or to a new results page

            uploaded_file = form.cleaned_data['file']
            file_data = ""

            for chunk in uploaded_file.chunks():
                file_data += chunk.decode('utf-8')

            answer = answer_single_question(question=form.cleaned_data['question'],
                                            dataset=file_data)

            return render(request, "experiment_app/results.html",
                          {"answer": answer})  # this is a new html page that you have to create to show the results.

    else:
        form = TextFileUploadForm()
    return render(request, 'experiment_app/model_form_upload.html', {
        'form': form
    })
