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
from common.segmentation import Foram

class BasicUploadView(View):
    def get(self, request):
        imgLocations = request.GET.getlist('imgLocation', None)
        if imgLocations:
            photos_list = []
            parent_list = []
            for imgLocation in imgLocations:
                # parent = ImgParent.objects.filter(imgLocation=imgLocation)[0]
                parent = ImgParent.objects.get(pk=int(imgLocation))
                parent.custom_url = parent.imgLocation.url[1:]
                parent_list.append(parent)
                child = Img.objects.get(parentImage=parent)
                child.custom_url = child.imgLocation.url[1:]
                photos_list.append(child)
                # tag = True
                # try:
                #     img = Img.objects.get(imgLocation=imgLocation)
                # except Img.DoesNotExist:
                #     tag = False
                # if tag == True:
                    # photos_list.append(img)
            return render(self.request, 'upload/imgUpload.html', {'photos': photos_list, 'parents': parent_list})
        else:
            return render(self.request, 'upload/imgUpload.html')

    def post(self, request):
        if 'edit_img_id' in request.POST: # editing the tags
            try:
                corrected_species = Species.objects.get(name=request.POST['species'])
            except Species.DoesNotExist:
                corrected_species = Species(name=request.POST['species'], total=1)
            corrected_species.save()
            Img.objects.filter(pk=request.POST['edit_img_id']).update(species=corrected_species)
            url = request.POST['original_url']
            return redirect(url)
        elif 'delete_img_id' in request.POST:
            img = Img.objects.get(pk=request.POST['delete_img_id'])
            img.imgLocation.delete()
            img.delete()
            url = request.POST['original_url']
            return redirect(url)
        else:   # upload the images
            foram_obj_list = []
            form = ImageUploadForm(request.POST, request.FILES)
            if form.is_valid():
                for file in request.FILES.getlist('imgLocation'):
                    foram_obj = Foram(file)
                    foram_obj.store_parents()
                    foram_obj.set_species()
                    foram_obj.store_children()
                    foram_obj_list.append(foram_obj)
                data = {'is_valid': True, 'urls': [obj.parent_obj.pk for obj in foram_obj_list]}
            else:
                print("erroraleart", form.errors)
                data = {'is_valid': False}
        return JsonResponse(data)


def showImg(request):
    if 'img_id' in request.POST:
        try:
            corrected_species = Species.objects.get(name=request.POST['species'])
        except Species.DoesNotExist:
            corrected_species = Species(name=request.POST['species'], total=1)
        corrected_species.save()
        Img.objects.filter(pk=request.POST['img_id']).update(species=corrected_species)
        url = request.POST['original_url']
        return redirect(url)
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

'''
Notes on what runs first
1. The else statement in post
2. The get method

'''