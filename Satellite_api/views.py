# views.py
from django.shortcuts import render
from .form import UploadImageForm
from .model_units import CustomModel, ImageProcess

class_labels = {0: 'cloudy', 1: 'desert', 2: 'green_area', 3: 'water'}
model_path = 'C://Users//oumaima//OneDrive//Documents//script//DL_app//DL_app//model//VGG16.keras'
def prediction(request):
    return render(request, 'prediction.html')

def home(request):
    if request.method == 'POST':

        form = UploadImageForm(request.POST, request.FILES)

        if form.is_valid():
            form = UploadImageForm(request.POST, request.FILES)

            uploaded_image = form.save()
            img_path = uploaded_image.satellite_image.path
            custom_model = CustomModel('VGG16', (224, 224, 3), 4)
            custom_model.load_weights(model_path)
            image_processor = ImageProcess(img_path)
            img_array = image_processor.process_image()

            predicted_class = image_processor.predict_class(custom_model, img_array)
            predicted_labels = [class_labels[index] for index in predicted_class.numpy()]

            print(f'The predicted class labels are: {predicted_labels}')

            context = {'form': form, 'uploaded_image': uploaded_image, 'predictions': predicted_labels}

            print(uploaded_image.satellite_image.url)
            return render(request, 'prediction.html',{'form': form})
    else:
        form = UploadImageForm()
    return render(request, 'prediction.html', {'form': form})
