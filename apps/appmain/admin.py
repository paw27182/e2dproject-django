from django.contrib import admin

from .models import DatabaseInfo, UserInfo

# Register your models here.

admin.site.register(DatabaseInfo)
admin.site.register(UserInfo)
