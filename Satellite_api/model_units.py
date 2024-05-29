from tensorflow.keras.preprocessing import image
import tensorflow as tf
import numpy as np
import cv2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dense, Dropout
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.applications import ResNet50, VGG16, EfficientNetB3
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization


class CustomModel(tf.keras.Model):
    def __init__(self, model_name, img_shape, class_count):
        super(CustomModel, self).__init__()
        self.model_name = model_name
        self.img_shape = img_shape
        self.class_count = class_count
        self.base_model = self.load_base_model()
        
        self.batch_norm = BatchNormalization()
        self.dense_1 = Dense(256, activation='relu')
        self.dropout_1 = Dropout(0.3)
        self.dense_2 = Dense(64, activation='relu')
        self.dropout_2 = Dropout(0.3)
        self.output_layer = Dense(self.class_count, activation='softmax')

    def load_base_model(self):
        if self.model_name == 'ResNet50':
            return ResNet50(include_top=False, weights="imagenet", input_shape=self.img_shape, pooling='max')
        elif self.model_name == 'VGG16':
            return VGG16(include_top=False, weights="imagenet", input_shape=self.img_shape, pooling='max')
        elif self.model_name == 'EfficientNetB3':
            return EfficientNetB3(include_top=False, weights="imagenet", input_shape=self.img_shape, pooling='max')
        else:
            print("Invalid model name, exiting...")
            exit()

    def call(self, inputs):
        x = self.base_model(inputs)
        x = self.batch_norm(x)
        x = self.dense_1(x)
        x = self.dropout_1(x)
        x = self.dense_2(x)
        x = self.dropout_2(x)
        return self.output_layer(x)

class ImageProcess:
    def __init__(self, img_path, target_size=(224, 224)):
        self.img_path = img_path
        self.target_size = target_size

    def process_image(self):
        img = image.load_img(self.img_path, target_size=self.target_size)
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0
        return img_array

    def predict_class(self, model, img_array):
        predictions = model.predict(img_array)
        predicted_classes = tf.argmax(predictions, axis=1)
        print(f'The predicted class is: {predicted_classes}')
        label_to_class = {
            0: 'cloudy',
            1: 'desert',
            2: 'green_area',
            3: 'water'
            # Add more mappings if needed
        }

        # Map indices to labels
        predicted_labels = [label_to_class[index] for index in predicted_classes.numpy()]
        print(f'The predicted class labels are: {predicted_labels}')
        return predicted_classes