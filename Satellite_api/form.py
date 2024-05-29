# forms.py
from django import forms
from .models import Satellite

class UploadImageForm(forms.ModelForm):
    class Meta:
        model = Satellite
        fields = ['satellite_image']
