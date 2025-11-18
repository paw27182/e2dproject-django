from django.urls import path

from .views import appmain

urlpatterns = [
    path('appmain/', appmain),
]
