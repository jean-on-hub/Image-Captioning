# Generated by Django 4.0.3 on 2022-03-28 16:30

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('caption', '0002_image_model_caption'),
    ]

    operations = [
        migrations.AlterField(
            model_name='image_model',
            name='caption',
            field=models.TextField(default='caption'),
        ),
    ]
