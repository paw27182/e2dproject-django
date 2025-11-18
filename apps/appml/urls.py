from django.urls import path

from .views import appml

urlpatterns = [
    path('appml/', appml),
]
