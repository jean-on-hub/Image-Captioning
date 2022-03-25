from django import forms 
from .models import Image_model
  
# class ImageForm(forms.ModelForm): 
  
#     class Meta: 
#         model = Image_model
#         fields = ['name', 'image'] 
class ImageForm(forms.Form):
    image_field = forms.ImageField()