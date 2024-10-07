from sklearn.model_selection import train_test_split
import pandas as pd
import os
import tensorflow as tf
from keras import layers, models, Model, preprocessing
import numpy as np
import matplotlib.pyplot as plt
from keras_preprocessing.image import ImageDataGenerator
import cv2

df = pd.read_csv('/Users/gayusri/GeoCracker/src/samples.csv')
PROC_IMAGE_DIR = '/Users/gayusri/GeoCracker/data/processed/images'

df.columns = df.columns.str.strip()

if 'label' not in df.columns:
    raise KeyError("'label' column not found in the DataFrame. Please ensure it exists.")


image_data = []

for index, row in df.iterrows():
    image_path = os.path.join(PROC_IMAGE_DIR, row['filename'])
    image_data.append((image_path, row['label']))

train_data, val_data = train_test_split(image_data, test_size=0.2, random_state=42)

train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

val_datagen = ImageDataGenerator(rescale=1.0 / 255)

train_generator = train_datagen.flow_from_dataframe(
    pd.DataFrame(train_data, columns=['filename', 'label']),
    directory=PROC_IMAGE_DIR,
    x_col='filename',
    y_col='label',
    target_size=(224, 224),  
    class_mode='binary',     
    batch_size=32
)

val_generator = val_datagen.flow_from_dataframe(
    pd.DataFrame(val_data, columns=['filename', 'label']),
    directory=PROC_IMAGE_DIR,
    x_col='filename',
    y_col='label',
    target_size=(224, 224),
    class_mode='binary',
    batch_size=32
)


#VGG16 for Image Classification:

input_layer = layers.Input(shape =(224,224,3))

#CNN block 1
x = layers.Conv2D(filters =64, kernel_size =3, padding ='same', activation='relu')(input_layer)
x = layers.Conv2D(filters =64, kernel_size =3, padding ='same', activation='relu')(x)
x = layers.MaxPool2D(pool_size =2, strides =2, padding ='same')(x)
#Relu is used in VCG16 and since we are starting with binary classification, we can still use it for hidden layers
#CNN block 2
x = layers.Conv2D(filters =128, kernel_size =3, padding ='same', activation='relu')(x)
x = layers.Conv2D(filters =128, kernel_size =3, padding ='same', activation='relu')(x)
x = layers.MaxPool2D(pool_size =2, strides =2, padding ='same')(x)

#CNN block 3
x = layers.Conv2D(filters =256, kernel_size =3, padding ='same', activation='relu')(x) 
x = layers.Conv2D(filters =256, kernel_size =3, padding ='same', activation='relu')(x) 
x = layers.Conv2D(filters =256, kernel_size =3, padding ='same', activation='relu')(x) 
x = layers.MaxPool2D(pool_size =2, strides =2, padding ='same')(x)

#CNN block 4
x = layers.Conv2D(filters =512, kernel_size =3, padding ='same', activation='relu')(x)
x = layers.Conv2D(filters =512, kernel_size =3, padding ='same', activation='relu')(x)
x = layers.Conv2D(filters =512, kernel_size =3, padding ='same', activation='relu')(x)
x = layers.MaxPool2D(pool_size =2, strides =2, padding ='same')(x)

#CNN block 5
x = layers.Conv2D(filters =512, kernel_size =3, padding ='same', activation='relu')(x)
x = layers.Conv2D(filters =512, kernel_size =3, padding ='same', activation='relu')(x)
x = layers.Conv2D(filters =512, kernel_size =3, padding ='same', activation='relu')(x)
x = layers.MaxPool2D(pool_size =2, strides =2, padding ='same')(x)

x = layers.Flatten()(x) 
x = layers.Dense(units = 4096, activation ='relu')(x) 
x = layers.Dense(units = 4096, activation ='relu')(x) 
output_layer = layers.Dense(units = 1, activation ='sigmoid')(x)
# Sigmoid is used for binary classification [0,1]

model = Model(inputs=input_layer, outputs =output_layer)

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=10
)

#model.save('bollard_recognition_model.h5')