from django.shortcuts import render,redirect
from django.urls import reverse
from upload.models import Img, Species, ImgParent
from upload.forms import ImageUploadForm
from django.http import JsonResponse, HttpResponseRedirect
from django.views import View
from .forms import ImageUploadForm
from common import segmentation as seg
import os
from pathlib import Path
from urllib.parse import urlparse, parse_qs


class BasicUploadView(View):
    def get(self, request):
        imgLocations = request.GET.getlist('imgLocation', None)
        if imgLocations:
            photos_list = []
            for imgLocation in imgLocations:
                parent = ImgParent.objects.filter(imgLocation=imgLocation)[0]
                child = Img.objects.filter(parentImage=parent)
                for c in child:
                    photos_list.append(c)
            return render(self.request, 'upload/imgUpload.html', {'photos': photos_list})
        else:
            return render(self.request, 'upload/imgUpload.html')

    def post(self, request):
        if 'uploaded_img_id' in request.POST: # editing the tags
            try:
                corrected_species = Species.objects.get(name=request.POST['species'])
            except Species.DoesNotExist:
                corrected_species = Species(name=request.POST['species'], total=1)
            corrected_species.save()
            Img.objects.filter(pk=request.POST['uploaded_img_id']).update(species=corrected_species)
            url = request.POST['original_url']
            return redirect(url)
        else:   # upload the images
            photos = []
            form = ImageUploadForm(request.POST, request.FILES)
            if form.is_valid():
                for file in request.FILES.getlist('imgLocation'):
                    forams, parent = seg.get_all_forams(file)
                    img_parent = seg.store_to_db(parent, forams, 'dummy', './media')
                    photos.append(ImgParent.objects.get(pk=img_parent.pk))
                data = {'is_valid': True, 'urls': [photo.imgLocation.name for photo in photos]}
            else:
                print("erroraleart", form.errors)
                data = {'is_valid': False}
        return JsonResponse(data)


def showImg(request):
    imgs = Img.objects.all() # 从数据库中取出所有的图片路径
    context = {
        'imgs' : imgs
    }
    return render(request, 'upload/showImg.html', context)


def clear_database(request):
    for species in Species.objects.all():
        species.delete()
    for photo in Img.objects.all():
        photo.imgLocation.delete()
        photo.delete()
    return redirect('uploadImage')
