from django.db import models

class Img(models.Model):
    imgLocation = models.ImageField(upload_to='segment', default='None')
    species = models.CharField(max_length=30, blank=True, default='None')
    parentImage = models.CharField(max_length=100, default='None')

class ImgParent(models.Model):
    imgLocation = models.ImageField(upload_to='parent', default='None')

class Species(models.Model):
    name = models.CharField(max_length=30, default='None')
    total = models.PositiveIntegerField(null = True)
