from django.db import models

# Create your models here.
class Image_model(models.Model):
    name = models.CharField(max_length=50)
    image = models.ImageField(upload_to='image/')
    caption = models.TextField(default="caption")
    # detected_img = models.ImageField(upload_to='image/',blank=True)
    
    def __str__(self):
        return self.name
    # def delete(self,*args,**kwargs):
    #     self.image.delete()
    #     self.detected_img.delete()
    #     super().delete(*args,**kwargs)