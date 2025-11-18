from django.conf import settings
from django.db import models

# Create your models here.


"""
Database -- 1:N -- Group -- 1:N -- User
                     |
                    1:N
                     |
                   Form

"""


# database information
class DatabaseInfo(models.Model):
    dbname = models.TextField(unique=True)
    hostname = models.TextField(null=True)
    accessid = models.TextField(null=True)
    accesspwd = models.TextField(null=True)
    administrator1 = models.TextField()
    administrator2 = models.TextField(null=True)
    update_at = models.DateField(auto_now_add=True)
    modified_by = models.TextField()

    def __str__(self):
        return self.dbname


# group information
class GroupInfo(models.Model):
    groupname = models.TextField()
    dbname = models.TextField()
    username = models.TextField()
    rolename = models.TextField()
    update_at = models.DateField(auto_now_add=True)
    modified_by = models.TextField()

    def __str__(self):
        return self.groupname


# form information
class FormInfo(models.Model):
    formname = models.TextField(unique=True)
    groupname = models.TextField()
    update_at = models.DateField(auto_now_add=True)
    modified_by = models.TextField()

    def __str__(self):
        return self.formname


# form list display
class FormList(models.Model):
    formname = models.TextField(unique=True)
    form_size = models.IntegerField(default=0)
    groupname = models.TextField()
    form_meta_data = models.TextField()
    update_at = models.DateField(auto_now_add=True)
    modified_by = models.TextField()

    def __str__(self):
        return self.formname


# user information
class UserInfo(models.Model):
    username = models.TextField(unique=True)
    password = models.TextField()
    update_at = models.DateField(auto_now_add=True)
    modified_by = models.TextField()

    def __str__(self):
        return self.username
