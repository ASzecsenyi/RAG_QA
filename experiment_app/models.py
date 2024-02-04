from django.db import models


# Create your models here.
class Experiment(models.Model):
    name = models.CharField(max_length=100, blank=True, null=True)
    description = models.TextField(blank=True, null=True)

    class Meta:
        db_table = 'Experiment'


class ExperimentTextDocument(models.Model):
    file = models.FileField(upload_to='uploads/', blank=True, null=True)
    file_path = models.CharField(max_length=100, blank=True, null=True)
    question = models.TextField()
    experiment = models.ForeignKey(Experiment, on_delete=models.CASCADE)

    class Meta:
        db_table = 'ExperimentTextDocument'


class ExperimentChunker(models.Model):
    chunk_length = models.IntegerField(blank=True, null=True)
    sliding_window_size = models.FloatField(blank=True, null=True)
    chunker_name = models.CharField(max_length=100, blank=True, null=True)
    experiment = models.ForeignKey(Experiment, on_delete=models.CASCADE)

    class Meta:
        db_table = 'ExperimentChunker'


class ExperimentRanker(models.Model):
    top_k = models.IntegerField(blank=True, null=True)
    ranker_name = models.CharField(max_length=100, blank=True, null=True)
    experiment = models.ForeignKey(Experiment, on_delete=models.CASCADE)

    class Meta:
        db_table = 'ExperimentRanker'


class ExperimentQA(models.Model):
    model_name = models.CharField(max_length=100, blank=True, null=True)
    experiment = models.ForeignKey(Experiment, on_delete=models.CASCADE)

    class Meta:
        db_table = 'ExperimentQA'





