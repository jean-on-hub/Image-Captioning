from django.shortcuts import render




from caption.forms import Image_form
from django import forms

from . import caption_code

from .models import Image_model
from django.conf.urls.static import static
from django.views.generic.edit import FormView

from django.http import HttpRequest
request = HttpRequest()
from pathlib import Path
from PIL import Image
class takeImage(FormView):
    form_class = Image_form
    template_name = 'caption/index.html'
    def form_valid(self, form):
        context ={}
        
        obj = form.cleaned_data.get('image')
        
        caption=caption_code.detect(obj.file)
        caption =' '.join(caption)
        print(caption)
        path =Path('display/media/image')
        obj = Image.open(obj)
        obj.save("caption/static/caption/image.png")
        caption = caption[0:-5]
        context['cap'] = caption
        context['image'] = obj
        
        print(context)
        return render( request,'caption/after.html', context)



from django.views.generic import DetailView
class after(DetailView):
    model = Image_model
    template_name = "caption/after.html"
    context_object_name ='image'