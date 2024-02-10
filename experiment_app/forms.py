import re
from typing import Any, List
from django import forms
from django.db import transaction

from rgqa_project.settings import MEDIA_ROOT
from .models import Experiment, ExperimentTextDocument, ExperimentNewsQaDocument, ExperimentChunker, ExperimentRanker, ExperimentQA


class ExperimentForm(forms.ModelForm):
    components = {
        ExperimentTextDocument: {
            'file': forms.FileField,
            'file_path': forms.CharField,
            'question': forms.CharField,
            'textdoc_name': forms.CharField,
        },
        ExperimentNewsQaDocument: {
            'story_id': forms.CharField,
            'newsqa_name': forms.CharField,
        },
        ExperimentChunker: {
            'chunker_type': forms.ChoiceField,
            'chunk_length': forms.IntegerField,
            'sliding_window_size': forms.FloatField,
            'chunker_name': forms.CharField,
        },
        ExperimentRanker: {
            'ranker_type': forms.ChoiceField,
            'top_k': forms.IntegerField,
            'ranker_name': forms.CharField,
        },
        ExperimentQA: {
            'model_type': forms.ChoiceField,
            'model_name': forms.CharField,
        }
    }

    choice_fields = {
        'chunker_type': ['CharChunker', 'SentChunker'],
        'ranker_type': ['TfidfRanker', 'SentEmbeddingRanker', 'GuessSimilarityRanker'],
        'model_type': ['MistralQA', 'GptQA', 'LlamaQA'],
    }

    class Meta:
        model = Experiment
        fields = ('name', 'description',)
        labels = {
            'name': 'Experiment Name',
            'description': 'Description',
        }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cols = {'ExperimentTextDocument': 'Document',
                     'ExperimentChunker': 'Chunker',
                     'ExperimentRanker': 'Ranker',
                     'ExperimentQA': 'QA'}
        self.id_dict = {
            'ExperimentTextDocument': {},
            'ExperimentNewsQaDocument': {},
            'ExperimentChunker': {},
            'ExperimentRanker': {},
            'ExperimentQA': {},
        }

        self.most_recent_experiment = Experiment.objects.last()

        # Initialize experiment fields
        self.initial['name'] = self.most_recent_experiment.name if self.most_recent_experiment else ''
        self.initial['description'] = self.most_recent_experiment.description if self.most_recent_experiment else ''

        # Initialize component fields for both existing and new forms, assuming at least one set for new instances
        for component, component_fields in self.components.items():
            self.init_component_fields(component, component_fields)

        # print(self.id_dict)

    def init_component_fields(self, component, component_fields: dict[str, Any]):
        # get the components of the most recent experiment
        components = component.objects.filter(experiment=self.most_recent_experiment) if self.most_recent_experiment else []

        # if not len(components):
        #     components = [component()]

        for i, comp in enumerate(components):
            for field_name, field_type in component_fields.items():
                field_key = f'{component.__name__}_{field_name}_{i}'
                self.fields[field_key] = field_type(required=False)
                self.initial[field_key] = getattr(comp, field_name, None)

                if field_name == 'file':
                    self.fields[field_key].widget.attrs['accept'] = '.txt'
                if field_type == forms.ChoiceField:
                    self.fields[field_key].choices = [(choice, choice) for choice in self.choice_fields[field_name]]

            # add a tick box to delete the component
            self.fields[f'delete_{component.__name__}_{i}'] = forms.BooleanField(required=False, label='Delete')

            # create a dictionary of the ids of the components
            self.id_dict[f'{component.__name__}'][i] = comp.id

    @transaction.atomic
    def save_components(self, experiment, component_model, component_fields: dict[str, Any], create_new: bool):
        # print(component_model)

        form_component_ids = set([int(key.split('_')[-1]) for key in self.data if
                                  any([key.startswith(f'{component_model.__name__}_{field_name}_') for field_name in component_fields])])

        new_components = []
        updated_components = []

        for i in form_component_ids:
            if self.cleaned_data.get(f'delete_{component_model.__name__}_{i}'):
                component_model.objects.filter(id=self.id_dict[component_model.__name__].get(i)).delete()
                continue
            comp = None
            comp_id = self.id_dict[component_model.__name__].get(i)
            if comp_id and not create_new:
                comp = component_model.objects.filter(id=comp_id).first()
            if not comp:
                comp = component_model()
                new_components.append(comp)

            for field_name, field_type in component_fields.items():
                field_key = f'{component_model.__name__}_{field_name}_{i}'
                if field_name == 'file':
                    setattr(comp, field_name, self.cleaned_data[field_key])
                else:
                    setattr(comp, field_name, self.data[field_key])

            # reassign the components from the most recent experiment to the new experiment
            comp.experiment = experiment

            if comp.id and comp not in updated_components and comp not in new_components:
                updated_components.append(comp)

        component_model.objects.bulk_create(new_components)
        # print([field.name for field in component_model._meta.fields])
        component_model.objects.bulk_update(updated_components, fields=[field.name for field in component_model._meta.fields if field.name != 'id'])

    def save(self, commit=True, create_new=False):
        # save request.FILES content to the MEDIA_ROOT
        for file in self.files.values():
            with open(f'{MEDIA_ROOT}/{file.name}', 'wb') as destination:
                for chunk in file.chunks():
                    destination.write(chunk)

        experiment = super().save(commit=False)
        if commit:
            experiment.save()
            # self.save_m2m()  # Save many-to-many data if any

        for component, component_fields in self.components.items():
            self.save_components(experiment, component, component_fields, create_new)

        for text_document in experiment.experimenttextdocument_set.all():
            # if The 'file' attribute has a file associated with it
            if text_document.file:
                # print('file', text_document.file.path)

                text_document.file_path = text_document.file.path
                text_document.file = None
                text_document.save()
                pass
            with open(text_document.file_path, 'r') as file:
                text_document.file_content = file.read()
            text_document.save()

        if self.most_recent_experiment:
            self.most_recent_experiment.delete()

        return experiment

    def clean(self):
        cleaned_data = super().clean()

        # Iterate over self.files
        for key, file in self.files.items():
            # Add each file to cleaned_data
            cleaned_data[key] = file

        return cleaned_data

    def get_initial_components_data(self):
        initial_components_data = {'components': []}

        # Iterate through the id_dict to get IDs and types of existing components
        for component_type, ids in self.id_dict.items():
            for i, id in ids.items():
                component_data = {'type': component_type, 'data': {}, 'id': id}
                for name, field in self.fields.items():
                    # Make sure to only include fields for the current component ID, and not include file fields
                    if name.startswith(component_type) and name.endswith(str(i)) and ('file' not in name or 'file_path' in name):
                        # Extract the field name without the component type and ID
                        field_key = "_".join(name.split('_')[1:-1])
                        component_data['data'][field_key] = self.initial.get(name, '')
                initial_components_data['components'].append(component_data)

        return initial_components_data
