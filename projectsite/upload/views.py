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
    def get(self, request, imgLocation=None):
        #url = self.request.get_full_path()
        #print(urlparse(url).query)
        #print(parse_qs(urlparse(url).query))
        imgLocation = request.GET.get('imgLocation', False)
        if imgLocation != False:
            photos_list = Img.objects.filter(imgLocation=imgLocation)
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
            url = Img.objects.get(pk=request.POST['uploaded_img_id'])
            address = '?imgLocation=' + url.imgLocation.name
            return redirect(address)
            #return redirect('uploadImage')
            #form = ImageUploadForm(self.request.POST, instance=instance)
        else:
            form = ImageUploadForm(self.request.POST, self.request.FILES)
            if form.is_valid():
                photo = form.save()
            #print("myspecies:", form.species_name)
            #_, file_ext = os.path.splitext(photo.imgLocation.name)
            #img = Img(imgLocation=request.FILES.get('img'))
            #forams = seg.get_all_forams(request.FILES.get('img'))
            #store_to_db(img, forams, 'ruber', './media/', '.jpg')
            #show_forams = Img.objects.filter(parentImage=img)
            #img_location = []
            #for foram in show_forams:
            #    img_location.append(foram.imgLocation)
                data = {'is_valid': True, 'url': photo.imgLocation.name}
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
