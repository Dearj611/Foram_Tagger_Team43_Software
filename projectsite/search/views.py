from django.shortcuts import render
from django.http import HttpResponse
from upload.models import Img


def searchImage(request):
    return render(request, 'search/search.html')

def dummy(request):
    pass

def search(species, count):
    '''
    Allows users to search by species and set how many images
    they want displayed
    '''
    query = Img.objects.filter(species=str(species))
    if query.exists():
        img_location = []
        for ele in query:
            img_location.append(ele.imgLocation)
        return img_location[:count]
    else:
        return
