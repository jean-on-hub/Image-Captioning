from django.urls import path
from .views import takeImage,after
from . import views
app_name = 'caption'
from django.conf import settings
from django.conf.urls.static import static
from django.urls import re_path
from django.views.static import serve  

urlpatterns = [
    
    path('', takeImage.as_view(), name='index' ),
    path('',after.as_view(),name='after'),
    re_path(r'^media/(?P<path>.*)$', serve, {
            'document_root': settings.MEDIA_ROOT,
        }),
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT) +  static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)

from django.contrib.staticfiles.urls import staticfiles_urlpatterns
urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
urlpatterns += staticfiles_urlpatterns()