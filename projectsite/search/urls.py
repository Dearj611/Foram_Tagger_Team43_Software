from django.urls import path

from search.views import searchImage, dummy

urlpatterns = [
    path('', searchImage),
    path('dummy', dummy)
]