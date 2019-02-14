from django.shortcuts import render
from upload.models import Img
from django.http import HttpResponseRedirect

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
        return render(request, "search/searchImg.html",{"forams":img_location, "species":search_key})
    else:
        return HttpResponseRedirect('')