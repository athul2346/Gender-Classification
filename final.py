from PIL import Image
import os
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.python.keras import layers,models
from multiprocessing import Pool
from keras.preprocessing.image import ImageDataGenerator
# Create data generators with data augmentation for training and validation
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255.0,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    zoom_range=0.1,
    shear_range=0.1,
    fill_mode='nearest',
    featurewise_center=True,
    featurewise_std_normalization=True,
)

validation_datagen=ImageDataGenerator(rescale=1.0/255.0)
# Load your dataset and preprocess it as needed
train_data_dir='.\\archive\\Training\\'
vailidation_data_dir='.\\archive\\Validation\\'

batch_size=32
image_size=(224,224)

#Create generators for training and validation

train_generator=train_datagen.flow_from_directory(
    train_data_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='binary'
)

validation_generator=validation_datagen.flow_from_directory(
    vailidation_data_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='binary'
)
#Creation of neural network layers
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(1, activation='sigmoid')  # Binary classification (male or female)
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy']
)

# Train the model
model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples,
    epochs=10
)
#saving the model
model.save("gender_classification_fullmodel.h5")
