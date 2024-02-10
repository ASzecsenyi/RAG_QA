from django.db import models


# Create your models here.
class Experiment(models.Model):
    name = models.CharField(max_length=100, blank=True, null=True)
    description = models.TextField(blank=True, null=True)

    class Meta:
        db_table = 'Experiment'


class ExperimentTextDocument(models.Model):
    file = models.FileField(blank=True, null=True)
    file_path = models.CharField(max_length=100, blank=True, null=True)
    file_content = models.TextField(blank=True, null=True)
    question = models.TextField()
    textdoc_name = models.CharField(max_length=100, blank=True, null=True)
    experiment = models.ForeignKey(Experiment, on_delete=models.CASCADE)

    class Meta:
        db_table = 'ExperimentTextDocument'


class ExperimentNewsQaDocument(models.Model):
    story_id = models.CharField(max_length=100, blank=True, null=True)
    newsqa_name = models.CharField(max_length=100, blank=True, null=True)
    experiment = models.ForeignKey(Experiment, on_delete=models.CASCADE)

    class Meta:
        db_table = 'ExperimentNewsQaDocument'


class ExperimentChunker(models.Model):
    # drop down menu for chunker_type: CharChunker, SentChunker
    chunker_type = models.CharField(max_length=100, blank=True, null=True)
    chunk_length = models.IntegerField(blank=True, null=True)
    sliding_window_size = models.FloatField(blank=True, null=True)
    chunker_name = models.CharField(max_length=100, blank=True, null=True)
    experiment = models.ForeignKey(Experiment, on_delete=models.CASCADE)

    class Meta:
        db_table = 'ExperimentChunker'


class ExperimentRanker(models.Model):
    # drop down menu for ranker_type: TfidfRanker, SentEmbeddingRanker, GuessSimilarityRanker
    ranker_type = models.CharField(max_length=100, blank=True, null=True)
    top_k = models.IntegerField(blank=True, null=True)
    ranker_name = models.CharField(max_length=100, blank=True, null=True)
    experiment = models.ForeignKey(Experiment, on_delete=models.CASCADE)

    class Meta:
        db_table = 'ExperimentRanker'


class ExperimentQA(models.Model):
    model_type = models.CharField(max_length=100, blank=True, null=True)
    model_name = models.CharField(max_length=100, blank=True, null=True)
    experiment = models.ForeignKey(Experiment, on_delete=models.CASCADE)

    class Meta:
        db_table = 'ExperimentQA'





