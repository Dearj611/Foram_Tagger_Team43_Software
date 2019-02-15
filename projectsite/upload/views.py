from django.shortcuts import render,redirect
from upload.models import Img
from upload.forms import ImageUploadForm
from django.http import JsonResponse
from django.views import View
from .forms import ImageUploadForm
from common import segmentation as seg
import os
from pathlib import Path


class BasicUploadView(View):
    def get(self, request):
        photos_list = Img.objects.all()
        #photos_list = Img.objects.filter(parentImage=photo.imgLocation.name)
        return render(self.request, 'upload/imgupload.html', {'photos': photos_list})

    def post(self, request):
        if request.method == 'POST':
            form = ImageUploadForm(self.request.POST, self.request.FILES)
            if form.is_valid():
                photo = form.save()
                #_, file_ext = os.path.splitext(photo.imgLocation.name)
                #img = Img(imgLocation=request.FILES.get('img'))
                #forams = seg.get_all_forams(request.FILES.get('img'))
                #store_to_db(img, forams, 'ruber', './media/', '.jpg')
                #show_forams = Img.objects.filter(parentImage=img)
                #img_location = []
                #for foram in show_forams:
                #    img_location.append(foram.imgLocation)
                data = {'is_valid': True, 'name': photo.imgLocation.name, 'url': photo.imgLocation.url, 'species':photo.species}
                #data = {'is_valid': True, 'content':img_location}
                return JsonResponse(data)
            else:
                form = ImageUploadForm()
                return render(request, 'name.html', {'form': form})


def showImg(request):
    imgs = Img.objects.all() # 从数据库中取出所有的图片路径
    context = {
        'imgs' : imgs
    }
    return render(request, 'upload/showImg.html', context)


def clear_database(request):
    for photo in Img.objects.all():
        photo.imgLocation.delete()
        photo.delete()
    return redirect(request.POST.get('next'))
