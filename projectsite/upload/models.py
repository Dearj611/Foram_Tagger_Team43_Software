from django.db import models


class Img(models.Model):
    imgLocation = models.CharField(max_length=100, default='None')
    species = models.CharField(max_length=30, default='None')
    parentImage = models.CharField(max_length=100, default='None')