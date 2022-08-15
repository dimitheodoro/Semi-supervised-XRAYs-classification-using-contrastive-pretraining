from tensorflow.keras.models import load_model
import PIL
import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import tensorflow as tf
import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

# finetuning_model.load_weights('/content/drive/MyDrive/semi.h5')
finetuning_model.load_weights('/content/semi.h5')
labels = {0: 'Normal', 1: 'Pathological'}
img_size=224

def get_prediction(image_path,my_model,labels):        
        image_loaded = PIL.Image.open(image_path)
        image_loaded = image_loaded.resize((img_size, img_size))
        image_loaded = np.asarray(image_loaded)
      
        if len(image_loaded.shape) < 3:
          image_loaded = np.stack([image_loaded.copy()] * 3, axis=2)
        
        preprocessed_image = preprocess_input(image_loaded)
        preprocessed_image = np.expand_dims(preprocessed_image, axis=0)
        
        predictions=my_model.predict(preprocessed_image)
        class_predicted = np.argmax(predictions[0])
        class_predicted_name = labels[class_predicted]                                  
        

        return class_predicted_name,predictions


def get_prediction2(image_loaded,my_model,labels):        
        # image_loaded = PIL.Image.open(image_path)
        # image_loaded = image_loaded.resize((img_size, img_size))
        # image_loaded = np.asarray(image_loaded)
      
        if len(image_loaded.shape) < 3:
          image_loaded = np.stack([image_loaded.copy()] * 3, axis=2)
        
        preprocessed_image = preprocess_input(image_loaded)
        # preprocessed_image = np.expand_dims(preprocessed_image, axis=0)
        
        predictions=my_model.predict(preprocessed_image)
        class_predicted = np.argmax(predictions[0])
        class_predicted_name = labels[class_predicted]                                  
        

        return class_predicted_name,predictions

trial_image_path ='/content/PNEUMONIA(3419).jpg'
get_prediction(trial_image_path, finetuning_model,labels)

