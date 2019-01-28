from django import forms
from .models import Img

<<<<<<< HEAD
"""
class ImageUploadForm(forms.ModelForm):
    image = forms.ImageField()
"""

class ImageUploadForm(forms.ModelForm):
    class Meta:
        model = Img
        fields = ('imgLocation', )
=======
class ImageUploadForm(forms.Form):
    '''
    Image upload form
    '''
    image = forms.ImageField()
>>>>>>> master
