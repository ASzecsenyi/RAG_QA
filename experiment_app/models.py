from django.db import models


# Create your models here.


class TextFileUpload(models.Model):
    file = models.FileField(upload_to='uploads/', blank=True, null=True)
    file_path = models.CharField(max_length=100, blank=True, null=True)
    question = models.TextField()






