from django.db import models

# Create your models here.
class Foram(models.Model):
    image = models.ImageField()
    species = models.CharField(max_length=50)
