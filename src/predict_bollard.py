import tensorflow as tf
from keras import preprocessing
import numpy as np
import cv2
import matplotlib.pyplot as plt

#model = tf.keras.models.load_model('bollard_recognition_model.h5')

def highlight_image(image_path, model):
    img = preprocessing.image.load_img(image_path, target_size=(224, 224))  
    img_array = preprocessing.image.img_to_array(img) 
    img_array = np.expand_dims(img_array, axis=0) 
    img_array = img_array / 255.0 

    prediction = model.predict(img_array)[0][0]

    original_image = cv2.imread(image_path)
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB) 

    label = "Austrian Bollard" if prediction > 0.5 else "Not Austrian Bollard"
    confidence = prediction if prediction > 0.5 else 1 - prediction

    height, width, _ = original_image.shape
    start_point = (int(width * 0.25), int(height * 0.25))
    end_point = (int(width * 0.75), int(height * 0.75))  

    color = (0, 255, 0) if prediction > 0.5 else (255, 0, 0)  
    thickness = 3
    cv2.rectangle(original_image, start_point, end_point, color, thickness)

    font = cv2.FONT_HERSHEY_SIMPLEX
    text = f"{label} ({confidence:.2f})"
    font_scale = 1
    font_thickness = 2
    text_color = (0, 255, 0) if prediction > 0.5 else (255, 0, 0)
    text_position = (int(width * 0.05), int(height * 0.95))

    cv2.putText(original_image, text, text_position, font, font_scale, text_color, font_thickness, cv2.LINE_AA)

    plt.figure(figsize=(6, 6))
    plt.imshow(original_image)
    plt.axis('off')
    plt.show()

    return label

image_path = '/Users/gayusri/GeoCracker/data/processed/images/bollard_89.jpg'
label = highlight_image(image_path, model)
print(f"The model predicts: {label}")