from django.db import models

class ImgParent(models.Model):
    '''
    I created this model so that I could use it for testing the accuracy
    of my image segmentation algorithm
    i.e. record how many forams are actually in this image
    '''
    imgLocation = models.ImageField(upload_to='parent', default='None')


class Species(models.Model):
    '''
    Will include some stats here later, easier to add columns than to
    remove them
    '''
    name = models.CharField(max_length=30, primary_key=True)
    total = models.PositiveIntegerField(null=True)

class Img(models.Model):
    imgLocation = models.ImageField(upload_to='', default='None')
    species = models.ForeignKey(Species, on_delete=models.CASCADE, null=True)
    parentImage = models.ForeignKey(ImgParent, on_delete=models.CASCADE, null=True)




'''
on_delete=CASCADE means that if I delete a species, then all the records
in the Img table referencing that species is deleted
i.e. you delete species = 'G. hazta', then all the records that reference
that G. hazta is deleted

on_delete=PROTECT
say your foreign key referenced a look-up table. 
if entries in that look-up table were deleted, this would records using
those entries to be deleted
If someone were to delete the gender "Female" from my Gender table, 
I CERTAINLY would NOT want that to instantly delete any and all people
 I had in my Person table who had that gender.

[('G. scitula', 123),
 ('G. truncatulinoides', 146),
 ('G. ruber pink', 12),
 ('G. sacculifer', 157),
 ('G. ruber', 141),
 ('N. humerosa', 103),
 ('G. crassaformis', 76),
 ('G. tumida', 153),
 ('G. elongatus', 58),
 ('N. acostaensis', 87),
 ('P. obliquiloculata', 98),
 ('G. siphonifera', 107),
 ('S. dehiscen', 120),
 ('G. ungulata', 70),
 ('G. hexagonus', 8),
 ('O. universa', 28),
 ('N. dutertrei', 50)]

17 species (for now)
say your foreign key referenced a look-up table.
if entries in that look-up table were deleted, this would records using
those entries to be deleted
If someone were to delete the gender "Female" from my Gender table,
I CERTAINLY would NOT want that to instantly delete any and all people
I had in my Person table who had that gender.
'''