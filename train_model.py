import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# Paths
DATASET_PATH = 'C:/Users/rohit/OneDrive/Desktop/DaTaSeT 2/'  # Make sure this is correct!

# Image settings
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10
NUM_CLASSES = 6  # Change this to 6

def create_model(input_shape, num_classes):
    model = Sequential([
        Conv2D(32, (3,3), activation='relu', input_shape=input_shape),
        MaxPooling2D(2,2),
        
        Conv2D(64, (3,3), activation='relu'),
        MaxPooling2D(2,2),
        
        Conv2D(128, (3,3), activation='relu'),
        MaxPooling2D(2,2),
        
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')  # Use num_classes here
    ])
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def main():
    # 1. Prepare ImageDataGenerators
    datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

    train_generator = datagen.flow_from_directory(
        DATASET_PATH,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training',
        shuffle=True
    )

    val_generator = datagen.flow_from_directory(
        DATASET_PATH,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation',
        shuffle=False
    )

    # 2. Create Model
    input_shape = (IMAGE_SIZE[0], IMAGE_SIZE[1], 3)
    # num_classes = train_generator.num_classes # No longer needed, we set it globally

    model = create_model(input_shape, NUM_CLASSES)

    # 3. Train Model
    model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=EPOCHS
    )

    # 4. Save Model
    model.save('medical_image_classifier.h5')
    print("✅ Model saved as medical_image_classifier.h5")

if __name__ == "__main__":
    main()
