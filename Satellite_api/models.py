from django.db import models

class Satellite(models.Model):
    satellite_image =  models.ImageField(null=True) 
