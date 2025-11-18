from django.urls import path

# from django.contrib.auth.views import LoginView, LogoutView
from .views import changepassword_view, index_view, login_view, logout_view, signup_view

# from .views import userfunc


app_name = 'accounts'

urlpatterns = [
    path('', index_view),
    path('login/', login_view, name='login'),

    # path('logout/', LogoutView.as_view(), name='logout'),
    path('logout/', logout_view, name='logout'),

    path('signup/', signup_view, name='signup'),
    path('changepassword/', changepassword_view, name='changepassword'),

    # path('userfunc/', userfunc, name='userfunc'),
]
