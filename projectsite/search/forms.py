from django import forms

class SpeciesForm(forms.Form):
    #Just for illustrative purposes, remove this later if you want
    species = forms.CharField(label='Name of Species', max_length=50)
    number = forms.IntegerField(label='No. of Images')