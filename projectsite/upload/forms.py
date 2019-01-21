from django import forms
from .models import Img

"""
class ImageUploadForm(forms.ModelForm):
    image = forms.ImageField()
"""

class ImageUploadForm(forms.ModelForm):
    class Meta:
        model = Img
        fields = ('imgLocation', )
