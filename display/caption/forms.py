# from django import forms 
# from .models import Image_model
  
# # class ImageForm(forms.ModelForm): 
  
# #     class Meta: 
# #         model = Image_model
# #         fields = ['name', 'image'] 
# class ImageForm(forms.Form):

#     image = forms.ImageField()
#     name = forms.CharField(max_length=50)
#     # class Meta: 
#     #     model = Image_model
#     #     fields = ['image']
#     def get_data(self):
#         pass
from django import forms
from  .models import Image_model
class Image_form(forms.Form):
    image = forms.ImageField() 