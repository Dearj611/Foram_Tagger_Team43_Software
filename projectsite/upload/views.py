from django.shortcuts import render, redirect
from upload.models import Img, Species, ImgParent
from upload.forms import ImageUploadForm
from django.http import JsonResponse
from django.views import View
from common.segmentation import Foram
from azure.storage.blob import BlockBlobService
import os
import tempfile

class BasicUploadView(View):
    def get(self, request):
        imgLocations = request.GET.getlist('imgLocation', None)
        if imgLocations:
            photos_list = []
            parent_list = []
            for imgLocation in imgLocations:
                parent = ImgParent.objects.get(pk=int(imgLocation))
                parent.custom_url = parent.imgEdited.url[1:]
                parent_list.append(parent)
                child = Img.objects.filter(parentImage=parent)
                temp_list=[]
                for c in child:
                    c.custom_url = c.imgLocation.url[1:]
                    temp_list.append(c)
                photos_list.append(temp_list)
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
            Foram.update_species(request.POST['edit_img_id'], request.POST['species'])
            url = request.POST['original_url']
            return redirect(url)
        elif 'delete_img_id' in request.POST:
            Foram.delete_foram(request.POST['delete_img_id'])
            url = request.POST['original_url']
            return redirect(url)
        else:   # upload the images
            foram_obj_list = []
            form = ImageUploadForm(request.POST, request.FILES)
            if form.is_valid():
                with tempfile.TemporaryDirectory() as dirpath:
                    for file in request.FILES.getlist('imgLocation'):
                        foram_obj = Foram(file, dirpath)
                        foram_obj.segment()
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