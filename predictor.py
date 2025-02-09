import tensorflow as tf
from keras.layers import TFSMLayer
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

filepath = r"PATH"  #Change to the path of the image you want to classify

model_layer = TFSMLayer(r"Saved_Models\Neuro93", call_endpoint="serving_default")

classes = ["Glioma","Meningioma","No Tumor", "Pituitary"]

def get_image(filepath):    #Processes image for clasification
    image = cv2.imread(filepath)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224, 224))
    image = image.astype(np.float32) 
    image = np.expand_dims(image, axis=0)
    return image

def predict_class(filepath):   #Runs image through model to get prediction
    image = get_image(filepath)
    output = model_layer(image)
    predictions_array = output['output_0'].numpy()[0]
    predicted_class = classes[predictions_array.argmax()]
    confidence = max(predictions_array)*100
    return predicted_class, confidence, predictions_array


prediction = predict_class(filepath) 

image = mpimg.imread(filepath) 
plt.title(f'Classification: {prediction[0]}  |  Confidence: {prediction[1]:.2f}')
plt.imshow(image)
plt.show()




