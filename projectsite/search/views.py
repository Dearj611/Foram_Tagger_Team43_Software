from django.shortcuts import render
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
