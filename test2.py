import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Model


tf.compat.v1.disable_eager_execution()
# Define paths to your dataset directories
base_path = 'D:\\project-96'
face_mask_path = os.path.join(base_path, 'face_data')  # Assuming correct directory

# Define image size and batch size
image_size = (224, 224)
batch_size = 32

# Create data generators
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

# Assuming you have a combined generator for all tasks
train_gen = datagen.flow_from_directory(
    face_mask_path,
    target_size=image_size,
    batch_size=batch_size,
    subset='training',
    class_mode='binary'  # Adjust class_mode based on your data
)

val_gen = datagen.flow_from_directory(
    face_mask_path,
    target_size=image_size,
    batch_size=batch_size,
    subset='validation',
    class_mode='binary'  # Adjust class_mode based on your data
)

# Repeat the same for other datasets (helmet, seat_belt, lane)

def create_multi_task_model():
    # Backbone Network
    backbone = EfficientNetB0(include_top=False, input_shape=(224, 224, 3))
    x = Flatten()(backbone.output)

    # Task 1: Face Mask Detection
    face_mask_output = Dense(1, activation='sigmoid', name='face_mask')(x)

    # Task 2: Helmet Detection
    helmet_output = Dense(1, activation='sigmoid', name='helmet')(x)

    # Task 3: Seat Belt Detection
    seat_belt_output = Dense(1, activation='sigmoid', name='seat_belt')(x)

    # Task 4: Lane Detection (assuming 80 classes for lanes)
    lane_output = Dense(80, activation='softmax', name='lane')(x)

    # Create model
    model = Model(inputs=backbone.input, outputs=[face_mask_output, helmet_output, seat_belt_output, lane_output])

    # Compile model with a multi-task loss
    model.compile(optimizer='adam',
                  loss={'face_mask': 'binary_crossentropy',
                        'helmet': 'binary_crossentropy',
                        'seat_belt': 'binary_crossentropy',
                        'lane': 'categorical_crossentropy'},
                  metrics={'face_mask': 'accuracy',
                           'helmet': 'accuracy',
                           'seat_belt': 'accuracy',
                           'lane': 'accuracy'})
    return model

multi_task_model = create_multi_task_model()

# Assuming you've addressed the data generator issues
# and have combined generators for all tasks

# Train the model
multi_task_model.fit(
    train_gen,
    epochs=50,
    validation_data=val_gen
)

# Save the trained model
multi_task_model.save('D:\\project-96\\unified_model.h5')

from tensorflow.keras.models import load_model

# Load the model
model = load_model('D:\\project-96\\unified_model.h5')

# Make predictions
# img = preprocess_image('path_to_your_image')
# predictions = model.predict(img)
