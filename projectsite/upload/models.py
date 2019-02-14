from django.db import models
class ImgParent(models.Model):
    imgLocation = models.ImageField(upload_to='parent', default='None')

class Species(models.Model):
    name = models.CharField(max_length=30, default='None')
    total = models.PositiveIntegerField(null=True)

class Img(models.Model):
    imgLocation = models.ImageField(upload_to='segment', default='None')
    species = models.ForeignKey(Species, on_delete=models.CASCADE)
    parentImage = models.ForeignKey(ImgParent, on_delete=models.CASCADE)
