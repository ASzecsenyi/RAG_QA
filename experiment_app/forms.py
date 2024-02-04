from typing import Any, List
from django import forms
from django.db import transaction

from .models import Experiment, ExperimentTextDocument, ExperimentChunker, ExperimentRanker, ExperimentQA


class ExperimentForm(forms.ModelForm):
    components = {
        ExperimentTextDocument: {
            'file': forms.FileField,
            'question': forms.CharField,
        },
        ExperimentChunker: {
            'chunk_length': forms.IntegerField,
            'sliding_window_size': forms.FloatField,
            'chunker_name': forms.CharField,
        },
        ExperimentRanker: {
            'top_k': forms.IntegerField,
            'ranker_name': forms.CharField,
        },
        ExperimentQA: {
            'model_name': forms.CharField,
        }
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
        self.id_dict = {}
        # Initialize component fields for both existing and new forms, assuming at least one set for new instances
        for component, component_fields in self.components.items():
            self.init_component_fields(component, component_fields)

    def init_component_fields(self, component, component_fields: dict[str, Any]):
        # get the most recent experiment
        most_recent_experiment = Experiment.objects.last()
        # get the components of the most recent experiment
        components = component.objects.filter(experiment=most_recent_experiment) if most_recent_experiment else None

        if not components:
            components = [component()]

        for i, comp in enumerate(components):
            for field_name, field_type in component_fields.items():
                field_key = f'{field_name}_{i}'
                self.fields[field_key] = field_type(required=False)
                self.initial[field_key] = getattr(comp, field_name, None)

                if field_name == 'file':
                    self.fields[field_key].widget.attrs['accept'] = '.txt'
                    # add file_path string to the form and make it uneditable
                    self.fields[f'file_path_{i}'] = forms.CharField(required=False, widget=forms.TextInput(attrs={'readonly': 'readonly'}))
                    self.initial[f'file_path_{i}'] = comp.file_path

            # Add an id field for tracking updates
            self.id_dict[f'{component.__name__}_id_{i}'] = comp.id

    @transaction.atomic
    def save_components(self, experiment, component_model, component_fields: dict[str, Any]):
        if 'file' in component_fields:
            # add file_path to component_fields
            component_fields['file_path'] = forms.CharField
        indices = set(int(field_name.split('_')[-1]) for field_name in self.cleaned_data if
                      any(field_name.startswith(key) for key in component_fields.keys()))

        existing_ids = {c.id for c in component_model.objects.filter(experiment=experiment)}

        to_create: List[component_model] = []
        to_update: List[component_model] = []

        for i in indices:
            defaults = {}
            for field_name in component_fields.keys():
                field_value = self.cleaned_data.get(f'{field_name}_{i}', None)
                if field_value is not None:
                    defaults[field_name] = field_value

            if not defaults:
                continue

            print(defaults)

            obj_id = self.id_dict.get(f"{component_model.__name__}_id_{i}", None)
            if obj_id and obj_id in existing_ids:
                obj = component_model.objects.get(id=obj_id)
                for attr, value in defaults.items():
                    setattr(obj, attr, value)
                to_update.append(obj)
            else:
                to_create.append(component_model(experiment=experiment, **defaults))

        if to_create:
            component_model.objects.bulk_create(to_create)

        if to_update:
            update_fields = list(component_fields.keys())
            component_model.objects.bulk_update(to_update, update_fields)

    def save(self, commit=True):
        experiment = super().save(commit=False)
        if commit:
            experiment.save()
            # self.save_m2m()  # Save many-to-many data if any

        for component, component_fields in self.components.items():
            self.save_components(experiment, component, component_fields)

        for text_document in experiment.experimenttextdocument_set.all():
            # if The 'file' attribute has a file associated with it
            if text_document.file:
                # set file_path to path to the file
                text_document.file_path = text_document.file.path
                # delete text_document.file
                text_document.file = None
                text_document.save()

        return experiment
