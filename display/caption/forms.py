from django import forms
from  .models import Image_model
class Image_form(forms.Form):
    image = forms.ImageField() 