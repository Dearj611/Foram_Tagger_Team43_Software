from django.shortcuts import render
from upload.models import Img
from django.http import HttpResponseRedirect
from django.http import HttpResponse
from django.core.paginator import Paginator, EmptyPage, PageNotAnInteger

def search(request):
    if 'search' in request.GET:
        search_key = request.GET.get('search')
        foram_list = Img.objects.filter(species=str(search_key)).order_by('id')
        if len(foram_list) == 0:
            message = "Sorry, there is no image matching this foram."
            return render(request,"search/searchImg.html",{"message":message})
    else:
        foram_list = Img.objects.get_queryset().order_by('id')
    page = request.GET.get('page', 1)
    paginator = Paginator(foram_list, 20)
    forams = paginator.page(page)
    return render(request,"search/searchImg.html",{"forams": forams})