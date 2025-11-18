from django.urls import path

from .views import downloads_view, topview_view

urlpatterns = [
    path('topview/', topview_view, name='topview'),
    path('downloads/<str:filename>', downloads_view, name='downloads'),
]
