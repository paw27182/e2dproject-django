from django.contrib import admin
from django.urls import include, path

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', include('accounts.urls')),
    path('', include('apps.topview.urls')),
    path('', include('apps.appmain.urls')),
    path('', include('apps.appml.urls')),
]
