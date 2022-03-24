from django.shortcuts import render

# Create your views here.
from django.http import HttpResponse


def index(request):
    return render(request, 'caption/index.html')
def after(request, data):
    return render(request, 'caption/after.html',data)