from django.shortcuts import render
from upload.models import Img

def search(request):
    if request.method == "GET":
        search_key =  request.GET.get('search')
        try:
            result = Img.objects.filter(species=str(search_key)) # filter returns a list so you might consider skip except part
        except Img.DoesNotExist:
            result = None
        img_location = []
        for foram in result:
            img_location.append(foram.imgLocation)
        return render(request,"search/searchImg.html",{"forams":img_location, "species":search_key})
    else:
        return HttpResponseRedirect('')

from django.http import HttpResponse

'''
def search(species, count):

    Allows users to search by species and set how many images
    they want displayed

    query = Img.objects.filter(species=str(species))
    if query.exists():
        img_location = []
        for ele in query:
            img_location.append(ele.imgLocation)
        return img_location[:count]
    else:
        return
'''
