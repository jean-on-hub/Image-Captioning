# Generated by Django 4.0.3 on 2022-03-28 16:24

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('caption', '0001_initial'),
    ]

    operations = [
        migrations.AddField(
            model_name='image_model',
            name='caption',
            field=models.TextField(default=None),
        ),
    ]
