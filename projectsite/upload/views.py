from django.http import HttpResponseRedirect
from django.shortcuts import render
from django.http import HttpResponse

from .forms import ImageUploadForm


def uploadImage(request):
    # if this is a POST request we need to process the form data
    if request.method == 'POST':
        # create a form instance and populate it with data from the request:
        form = ImageUploadForm(request.POST)
        # check whether it's valid:
        if form.is_valid():
            print('is_valid')
            # process the data in form.cleaned_data as required
            # ...
            # redirect to a new URL:
            return HttpResponseRedirect('/thanks/')

    # if a GET (or any other method) we'll create a blank form
    else:
        print('GET')
        form = ImageUploadForm()

    return render(request, 'upload/index.html', {'form': form})