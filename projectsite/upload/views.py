from django.shortcuts import render
from upload.models import Img
from upload.forms import ImageUploadForm
from django.http import JsonResponse
from django.views import View

class BasicUploadView(View):
    def get(self, request):
        photos_list = Img.objects.all()
        return render(self.request, 'upload/imgupload.html', {'photos': photos_list})

    def post(self, request):
        form = ImageUploadForm(self.request.POST, self.request.FILES)
        if form.is_valid():
            photo = form.save()
            data = {'is_valid': True, 'name': photo.imgLocation.name, 'url': photo.imgLocation.url}
        else:
            data = {'is_valid': False}
        return JsonResponse(data)

"""
def uploadImg(request): # 图片上传函数
    if request.method == 'POST':
        img = Img(imgLocation=request.FILES.get('img'))
        img.save()
    return render(request, 'upload/imgupload.html')
    # return HttpResponseRedirect('/thanks/')

    if request.method == 'GET':
        photos_list = Img.objects.all()
        return render(self.request, 'upload/imgupload.html', {'photos': photos_list})
"""
def showImg(request):
    imgs = Img.objects.all() # 从数据库中取出所有的图片路径
    context = {
        'imgs' : imgs
    }
    return render(request, 'upload/showImg.html', context)
