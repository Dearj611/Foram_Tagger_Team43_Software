from django.urls import path
from django.conf.urls.static import static
from django.conf import settings
from upload.views import uploadImg, showImg

urlpatterns = [
    path('', uploadImg, name='uploadImage'),
    path('show/', showImg, name='displayImage'),
]+ static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
