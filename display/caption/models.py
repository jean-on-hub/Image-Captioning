from django.db import models

# Create your models here.
class Image_model(models.Model):
    name = models.CharField(max_length=50)
    image = models.ImageField(upload_to='image/')
    caption = models.TextField(default="caption")
    
    
    def __str__(self):
        return self.name
    