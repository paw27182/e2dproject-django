import mimetypes
from pathlib import Path

from django.conf import settings
from django.contrib.auth.decorators import login_required
from django.http.response import HttpResponse
from django.shortcuts import render

# Create your views here.


@login_required()
# @login_required(login_url='/accounts/login/')
def topview_view(request):
    headline = "Excel to Database Application"
    message = f"Login ID: {request.user.email}"  # .username

    return render(request,
                  'topview/topview.html',
                  {'headline': headline,
                   'message': message},
                  )


def downloads_view(request, filename=''):
    # Define Django project base directory
    BASE_DIR = settings.BASE_DIR

    # filepath = BASE_DIR + '/download/' + filename
    filepath = Path(BASE_DIR, filename)

    # Open the file for reading content
    path = open(filepath, 'rb')

    # Set the mime type
    mime_type, _ = mimetypes.guess_type(filepath)

    # Set the return value of the HttpResponse
    response = HttpResponse(path, content_type=mime_type)

    # Set the HTTP header for sending to browser
    response['Content-Disposition'] = "attachment; filename=%s" % filename

    # Return the response value
    return response
