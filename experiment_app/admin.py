from django.contrib import admin

# Register your models here.

from .models import Experiment, ExperimentChunker, ExperimentQA, ExperimentRanker, ExperimentTextDocument, \
    ExperimentNewsQaDocument


@admin.register(Experiment)
class ExperimentAdmin(admin.ModelAdmin):
    list_display = ['name', 'id']
    search_fields = ['name']
    list_filter = []


@admin.register(ExperimentTextDocument)
class ExperimentTextDocumentAdmin(admin.ModelAdmin):
    list_display = ['id', 'experiment', 'question', 'file', 'file_path']
    search_fields = ['experiment__name']
    list_filter = ['experiment']


@admin.register(ExperimentNewsQaDocument)
class ExperimentNewsQaDocumentAdmin(admin.ModelAdmin):
    list_display = ['id', 'experiment', 'story_id']
    search_fields = ['experiment__name']
    list_filter = ['experiment']


@admin.register(ExperimentChunker)
class ExperimentChunkerAdmin(admin.ModelAdmin):
    list_display = ['chunker_type', 'chunker_name', 'id', 'experiment', 'chunk_length', 'sliding_window_size']
    search_fields = ['chunker_name', 'experiment__name']
    list_filter = ['experiment']


@admin.register(ExperimentRanker)
class ExperimentRankerAdmin(admin.ModelAdmin):
    list_display = ['ranker_type', 'ranker_name', 'id', 'experiment', 'top_k']
    search_fields = ['ranker_name', 'experiment__name']
    list_filter = ['experiment']


@admin.register(ExperimentQA)
class ExperimentQAAdmin(admin.ModelAdmin):
    list_display = ['model_type', 'model_name', 'id', 'experiment']
    search_fields = ['model_name', 'experiment__name']
    list_filter = ['experiment']
