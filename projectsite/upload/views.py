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
        #url = self.request.get_full_path()
        #print(urlparse(url).query)
        #print(parse_qs(urlparse(url).query))
        imgLocations = request.GET.getlist('imgLocation', None)
        if imgLocations:
            photos_list = []
            for imgLocation in imgLocations:
                photo = Img.objects.filter(imgLocation=imgLocation)[0]
                photos_list.append(photo)
            return render(self.request, 'upload/imgUpload.html', {'photos': photos_list})
        else:
            #print("didn't receive anything!")
            return render(self.request, 'upload/imgUpload.html')

    def post(self, request):
        if 'uploaded_img_id' in request.POST:
            try:
                corrected_species = Species.objects.get(name=request.POST['species'])
                #original_total = Species.objects.get(name=request.POST['species']).total
                #corrected_species.total = original_total + 1
            except Species.DoesNotExist:
                corrected_species = Species(name=request.POST['species'], total=1)
            corrected_species.save()
            Img.objects.filter(pk=request.POST['uploaded_img_id']).update(species=corrected_species)
            url = request.POST['original_url']
            return redirect(url)
            #return redirect('uploadImage')
            #form = ImageUploadForm(self.request.POST, instance=instance)
        else:
            print(request.FILES)
            print(request.POST)
            photos = []
            form = ImageUploadForm(request.POST, request.FILES)
            if form.is_valid():
                for file in request.FILES.getlist('imgLocation'):
                    print(file)
                    img = Img(imgLocation = file)
                    img.save()
                    photos.append(img)
                    print(photos)
            #print("myspecies:", form.species_name)
            #_, file_ext = os.path.splitext(photo.imgLocation.name)
            #img = Img(imgLocation=request.FILES.get('img'))
            #forams = seg.get_all_forams(request.FILES.get('img'))
            #store_to_db(img, forams, 'ruber', './media/', '.jpg')
            #show_forams = Img.objects.filter(parentImage=img)
            #img_location = []
            #for foram in show_forams:
            #    img_location.append(foram.imgLocation)
                data = {'is_valid': True, 'urls': [photo.imgLocation.name for photo in photos]}
                #return redirect('uploadImage')
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
