from django import forms
from .models import Img

"""
class ImageUploadForm(forms.ModelForm):
    image = forms.ImageField()
"""

class ImageUploadForm(forms.ModelForm):
    uploaded_img_id = forms.IntegerField(required=False)
    species_name = forms.CharField(required=False)
    class Meta:
        model = Img
        fields = ('imgLocation', 'species_name', 'uploaded_img_id')
