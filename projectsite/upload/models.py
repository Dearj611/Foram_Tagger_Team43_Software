from django.db import models


class ImgParent(models.Model):
    '''
    Will probably include statistics about each species in the future
    '''
    imgLocation = models.ImageField(upload_to='parent', default='None')


class Img(models.Model):
    imgLocation = models.ImageField(upload_to='segment', default='None')
    species = models.CharField(max_length=30, default='None')
    parentImage = models.ForeignKey(ImgParent, on_delete=models.CASCADE)
