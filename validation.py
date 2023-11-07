import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras

model = keras.models.load_model('gender_classification_fullmodel.h5')
output_male_dir = 'Male'
output_female_dir = 'Female'

os.makedirs(output_male_dir, exist_ok=True)
os.makedirs(output_female_dir, exist_ok=True)


def classify_and_save_images(input_folder):
    for filename in os.listdir(input_folder):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            img_path = os.path.join(input_folder, filename)

            # Load and preprocess the image
            img = cv2.imread(img_path)
            img = cv2.resize(img, (224, 224))  # Adjust the size as needed
            img = img / 255.0  # Normalize the image

            # Make a prediction
            img_array = np.expand_dims(img, axis=0)
            prediction = model.predict(img_array)[0][0]  # Get the single prediction value

            # Decide which folder to save in based on the prediction
            if prediction > 0.7:  # Adjust the threshold as needed
                output_path = os.path.join(output_male_dir, filename)
            else:
                output_path = os.path.join(output_female_dir, filename)

            # Move the image to the appropriate folder
            os.rename(img_path, output_path)


input_folder = '.\\image\\'


classify_and_save_images(input_folder)
