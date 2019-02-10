from django.urls import path
from django.conf.urls.static import static
from django.conf import settings
from upload.views import BasicUploadView, showImg, clear_database

urlpatterns = [
    path('clear/', clear_database, name='clear_database'),
    path('', BasicUploadView.as_view(), name='uploadImage'),
    path('show/', showImg, name='displayImage'),
]
