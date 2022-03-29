from django.shortcuts import render

# # Create your views here.
# from django.http import HttpResponse


# # def index(request):
# #     return render(request, 'caption/index.html')
# # def after(request, data):
# #     return render(request, 'caption/after.html',data)
# import numpy as np
# # from . import object_detection
# # import cv2


import os
from django.http import HttpResponseRedirect,HttpResponse
from django.urls import reverse_lazy
from django.views.generic import TemplateView
from caption.forms import Image_form
from django import forms

# from django.views.generic import DetailView
# # from djangoapp.models import Image_model

# # class Image(TemplateView):

# #     form = ImageForm
# #     template_name = 'image.html'

# #     def post(self, request, *args, **kwargs):

# #         form = ImageForm(request.POST, request.FILES)


# #         if form.is_valid():
# #             obj = form.save()
# #             print(obj.name)
# #             print(obj.image.url)
# #             outputFile,frame=object_detection.detect(obj.image.url)
# #             #outputFile=os.path.basename(outputFile)
# #             obj.detected_img = outputFile
# #             print(obj.detected_img)
# #             obj.save()
# #             return HttpResponseRedirect(reverse_lazy('image_display', kwargs={'pk': obj.id}))

# #         context = self.get_context_data(form=form)
# #         return self.render_to_response(context)

# #     def get(self, request, *args, **kwargs):
# #         return self.post(request, *args, **kwargs)

# # class ImageDisplay(DetailView):
# #     model = Image_model
# #     template_name = 'image_display.html'
# #     context_object_name = 'context'

# # def deleteimg(request,pk):
# #     if request.method=='POST':
# #         model = Image_model.objects.get(pk=pk)
# #         model.delete()
# #         return HttpResponseRedirect(reverse_lazy('home'))

from . import caption_code
# from .forms import ImageForm
from .models import Image_model
# from django.http import HttpResponseRedirect
# from django.urls import reverse_lazy
# from django.views.generic import TemplateView
# from caption.forms import ImageForm
from django.views.generic.edit import FormView
# from django.shortcuts import render_to_response
# # Create your views here.
from django.http import HttpRequest
request = HttpRequest()
class takeImage(FormView):
    form_class = Image_form
    template_name = 'caption/index.html'
    def form_valid(self, form):
        context ={}
        obj = form.cleaned_data.get('image')
        # context = self.get_context_data(form = form)
        # print(type(context['form']))
        caption=caption_code.detect(obj)
        caption =' '.join(caption)
        print(caption)
        context['cap'] = caption
        context['image'] = obj
        
        print(context)
        return render( request,'caption/after.html', context)

#        caption=caption_code.detect(FILES['image'])
#        obj = form.save(commit=False)
#        obj.caption = caption
#        obj.save()
#        context[obj] = obj
#        return render(request,'caption/after.html',context)
   
       
            
       
#     #    else:ageForm()
# #    context = {}
# #    def post(self, request, *args, **kwargs):
# #        context = {}
# #        form =ImageForm(request.POST,request.FILES)

#     #    return self.render_to_response(context)
# #    def after(self, context):
# #         # super.takeImage
# #         render('caption/after.html',context)
# #    def get(self, request, *args, **kwargs):
# #         return self.post(request, *args, **kwargs)
#     #    if request.method == "POST":
#     #     form = ImageForm(request.POST, request.FILES)

#     #     if form.is_valid():
#     #         name = 'image'
#     #         img = form.cleaned_data.get("image_field")
#     #         obj = Image_model.objects.create(
#     #                              image = img,
#     #                              name = name
#     #                              )
#     #         caption,image=caption_code.detect(obj.image)
#     #         obj.caption = caption
#     #         obj.save()
#     # else:
#     #     form = ImageForm()
#     # context['form']= form
#     # return render( request, "caption/index.html", context)
# 
from django.views.generic import DetailView
class after(DetailView):
    model = Image_model
    template_name = "caption/after.html"
    context_object_name ='image'