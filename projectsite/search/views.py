from django.shortcuts import render
from upload.models import Img
from django.http import HttpResponseRedirect
from django.http import HttpResponse


def search(request):
        if 'search' in request.GET:
            search_key = request.GET.get('search')
            try:
                result = Img.objects.filter(species=str(search_key)) # filter returns a list so you might consider skip except part
            except Img.DoesNotExist:
                result = None
            result = Img.objects.filter(species=str(search_key)) # filter returns a list so you might consider skip except part
            if not result.exists():
                message = "Sorry, there is no image matching this foram."
                return render(request,"search/searchImg.html",{"message":message})
            imgs = []
            for foram in result:
                imgs.append(foram)
        else:
            imgs = Img.objects.all()
        return render(request,"search/searchImg.html",{"forams":imgs})

