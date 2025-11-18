from pathlib import Path

import numpy as np
from django.contrib.auth import authenticate, get_user_model, login, logout
from django.http import HttpResponse, JsonResponse
from django.shortcuts import redirect, render
from django.urls import reverse
from django.views.decorators.csrf import csrf_exempt

# from django.contrib.auth import logout
# from django.views.generic import TemplateView

mid = Path(__file__).name  # module id

# Create your views here.

# @csrf_exempt
def index_view(request):
    return redirect("/login")  # slash


# @csrf_exempt
def login_view(request):
    if request.method == "POST":
        username = request.POST.get('username')
        password = request.POST.get('password')

        if authenticate(username=username, password=password):
            User = get_user_model()
            u = User.objects.get(email=username)
            # print(f"[{mid}] before {request.user.is_authenticated= }")
            login(request, u)
            # print(f"[{mid}] after  {request.user.is_authenticated= }")
            return redirect("/topview")  # slash
        else:
            return JsonResponse({"status": "NG", "message": "login failed."})

    else:  # GET
        headline = "Excel to Database Application"
        jumbotron_image = str(np.random.randint(1, 6, 1)[0]) + ".jpg"  # [1,5]
        return render(request,
                      'auth/login.html',
                      {'headline': headline,
                       'jumbotron_image': jumbotron_image},
                      )


@csrf_exempt
def signup_view(request):
    if request.method == "POST":
        pass
        # save into UserInfo, too.
        return JsonResponse({"status": "sorry, UNDISCLOSED"})
    else:  # GET
        return render(request, 'auth/signup.html')


@csrf_exempt
def changepassword_view(request):
    if request.method == "POST":
        pass
        return JsonResponse({"status": "sorry, UNDISCLOSED"})
    else:  # GET
        return render(request, 'auth/changepassword.html')


# @csrf_exempt
def logout_view(request):
    logout(request)
    return redirect('/login')


# def userfunc(request):
#     User = get_user_model()
#     target_email = "kate.walsh@example.com"
#     for obj in User.objects.filter(email=target_email).all():
#         print(obj.email)
#     return HttpResponse('<h1>hello world</h1>')
