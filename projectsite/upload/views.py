from django.http import HttpResponseRedirect
from django.shortcuts import render
from upload.models import Img

def uploadImg(request): # 图片上传函数
    if request.method == 'POST':
        img = Img(img_url=request.FILES.get('img'))
        img.save()
    return render(request, 'upload/imgupload.html')
    # return HttpResponseRedirect('/thanks/')

def showImg(request):
    imgs = Img.objects.all() # 从数据库中取出所有的图片路径
    context = {
        'imgs' : imgs
    }
    return render(request, 'upload/showImg.html', context)
