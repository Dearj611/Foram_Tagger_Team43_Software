from django.shortcuts import render
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
        #forams = seg.get_all_forams(request.FILES.get('img'))
        #photos_list = Img.objects.filter(parentImage=photo.imgLocation.name)
        return render(self.request, 'upload/imgupload.html', {'photos': photos_list})

    def post(self, request):
        form = ImageUploadForm(self.request.POST, self.request.FILES)
        if form.is_valid():
            photo = form.save()
            _, file_ext = os.path.splitext(photo.imgLocation.name)
            data = {'is_valid': True, 'name': photo.imgLocation.name, 'url': photo.imgLocation.url}
        else:
            data = {'is_valid': False}
        return JsonResponse(data)


def showImg(request):
    imgs = Img.objects.all() # 从数据库中取出所有的图片路径
    context = {
        'imgs' : imgs
    }
    return render(request, 'upload/showImg.html', context)
