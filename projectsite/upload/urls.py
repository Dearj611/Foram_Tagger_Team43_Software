from django.urls import path
from django.conf.urls.static import static
from django.conf import settings
from upload.views import uploadImg, showImg

urlpatterns = [
    path('', uploadImg, name='uploadImage'),
    path('show/', showImg, name='displayImage'),
]
