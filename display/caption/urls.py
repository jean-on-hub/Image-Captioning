from django.urls import path

from . import views
app_name = 'caption'

urlpatterns = [
    path('', views.index, name='index'),
    path('caption/after', views.after, name='after'),
]