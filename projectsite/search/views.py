from django.shortcuts import render
from upload.models import Img
from django.http import HttpResponseRedirect
from django.http import HttpResponse


def search(request):
    if 'search' in request.GET:
        imgs = []
        search_key = request.GET.get('search')
        result = Img.objects.filter(species=str(search_key))
        if len(result) == 0:
            message = "Sorry, there is no image matching this foram."
            return render(request,"search/searchImg.html",{"message":message})
        for foram in result:
            imgs.append(foram)
    else:
        imgs = Img.objects.all()
    return render(request,"search/searchImg.html",{"forams":imgs})