from django import forms
from .models import TextFileUpload


class TextFileUploadForm(forms.ModelForm):
    class Meta:
        model = TextFileUpload
        fields = ('file', 'file_path', 'question',)
        labels = {
            'file': 'Upload a file, or',
            'file_path': 'Enter a file path',
            'question': 'Ask a question'
        }


class ChunkerForm(forms.Form):
    chunk_length = forms.IntegerField()
    sliding_window_size = forms.FloatField()


class RankerForm(forms.Form):
    num_rank = forms.IntegerField()


class QAForm(forms.Form):
    model_name = forms.CharField()
