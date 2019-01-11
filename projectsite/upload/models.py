from django.db import models


class Img(models.Model):
    img_url = models.ImageField(upload_to='segment')
    species = models.CharField(max_length=30, default='None')
    parentImage = models.CharField(max_length=100, default='None')
