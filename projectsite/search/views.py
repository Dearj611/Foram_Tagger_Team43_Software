from django.shortcuts import render
<<<<<<< HEAD
from upload.models import Img

def search(request):
    if request.method == "GET":
        search_key =  request.GET.get('search')
        try:
            result = Img.objects.filter(species=search_key) # filter returns a list so you might consider skip except part
        except Img.DoesNotExist:
            result = None
        return render(request,"search/searchImg.html",{"forams":result})
    else:
        return render(request,"search/searchImg.html",{})
=======
from django.http import HttpResponse
from upload.models import Img


def index(request):
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
>>>>>>> master
