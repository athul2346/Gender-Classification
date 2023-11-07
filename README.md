Gender Classification

This Git repository contains code for classifying input images into male and female categories using a trained dataset. Two main Python scripts are provided for this task: final.py for training and validation.py for image classification.
Directory Structure

Before using the code, ensure that your directory has the following structure:
For Training (Using final.py):
- Your_Project_Directory/
  - archive/
    - training/
      - male/
        - [male images]
      - female/
        - [female images]
    - validation/
      - male/
        - [male images]
      - female/
        - [female images]
For Validation (Using validation.py):
- Your_Project_Directory/
  - image/
    - [input images to be classified]
final.py

final.py is responsible for training a model using TensorFlow. To use this script, follow these steps:
1)Install the required packages. You can use the following command to install the necessary packages:
    pip install tensorflow
2)Organize your training data as mentioned in the directory structure section.
3)Place the images in the respective folders (male and female).
4)Run the final.py script to train your model. It will create an h5 model as the output.

validation.py

validation.py is used for classifying input images using the trained model. To use this script:
1)Ensure that you have already trained the model using final.py and obtained the h5 model file.
2)Load the model in the script by specifying the correct path to the model file.
3)Place the images you want to classify in the image folder within your project directory.
4)Run the validation.py script. It will classify the input images into two folders, male and female, according to the trained model.

Feel free to customize the code to suit your specific needs. Good luck with your gender classification project!
