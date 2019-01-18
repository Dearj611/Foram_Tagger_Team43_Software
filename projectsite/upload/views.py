from django.http import HttpResponseRedirect
from django.shortcuts import render
from upload.models import Img
from .forms import ImageUploadForm
from common import segmentation as seg
import os
from pathlib import Path


def uploadImg(request): # 图片上传函数
    if request.method == 'POST':        
        forams = seg.get_all_forams(request.FILES.get('img'))
        _, file_ext = os.path.splitext(request.FILES.get('img').name)
        print(file_ext)
    return render(request, 'upload/imgUpload.html')
    # return HttpResponseRedirect('/thanks/')


def handle_uploaded_file(f):
    with open('some/file/name.txt', 'wb+') as destination:
        for chunk in f.chunks():
            destination.write(chunk)


def showImg(request):
    imgs = Img.objects.all() # 从数据库中取出所有的图片路径
    context = {
        'imgs' : imgs
    }
    return render(request, 'upload/showImg.html', context)
