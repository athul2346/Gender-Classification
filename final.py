import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras import layers,models
# from keras.preprocessing.image import ImageDataGenerator
# from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.python.keras.callbacks import EarlyStopping,ReduceLROnPlateau,ModelCheckpoint
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.callbacks import EarlyStopping,ReduceLROnPlateau,ModelCheckpoint
# from keras.applications import VGG16



# Load your dataset and preprocess it as needed
train_data_dir = 'D:\\Gender classification\\archive\\Training\\'
validation_data_dir = 'D:\\Gender classification\\archive\\Validation\\'

# Create data generators with data augmentation for training and validation
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
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

val_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1.0 / 255.0)
batch_size = 32
image_size = (224, 224)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='binary'
)

val_generator = val_datagen.flow_from_directory(
    validation_data_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='binary'
)

# Create a base model with transfer learning
base_model = tf.keras.applications.vgg16.VGG16(
    include_top=False,
    weights='imagenet',
    input_shape=(224, 224, 3)
)
base_model.trainable = False

# Define a custom layer for the VGG16 base model to implement get_config
class VGG16Wrapper(keras.layers.Layer):
    def __init__(self, base_model, **kwargs):
        super().__init__(**kwargs)
        self.base_model = base_model

    def call(self, inputs):
        return self.base_model(inputs)

    def get_config(self):
        config = super().get_config()
        config.update({"base_model": self.base_model})
        return config



# Wrap the VGG16 base model
base_model = VGG16Wrapper(base_model)

model = tf.keras.Sequential([
    base_model,
    layers.Flatten(),
    layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')
])

# Compile the model with custom learning rate schedule
initial_learning_rate = 0.001
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate, decay_steps=1000, decay_rate=0.9
)
optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

model.compile(optimizer=optimizer,
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Implement callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6)
model_checkpoint = ModelCheckpoint("best_model.h5", save_best_only=True)

# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    validation_data=val_generator,
    validation_steps=val_generator.samples // batch_size,
    epochs=30,
    callbacks=[early_stopping, reduce_lr, model_checkpoint]
)

model.save('final.h5')