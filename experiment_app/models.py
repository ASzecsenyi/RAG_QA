from django.db import models

# Create your models here.


class TextFileUpload(models.Model):
    file = models.FileField(upload_to='uploads/')
    question = models.TextField()
