import matplotlib
matplotlib.use('Agg') # FIX: Prevents the terminal from hanging
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import os

# Simplified paths relative to where this script is saved
TRAIN_DIR = 'dataset/train'
VAL_DIR = 'dataset/valid'

# 1. Loading Data
train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)

print(f"Loading images from {TRAIN_DIR}...")
# This will now find the folders correctly if you followed Step 1
train_gen = train_datagen.flow_from_directory(TRAIN_DIR, target_size=(224, 224), batch_size=32, class_mode='categorical')
val_gen = val_datagen.flow_from_directory(VAL_DIR, target_size=(224, 224), batch_size=32, class_mode='categorical')

# 2. Build Classifier Architecture
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(224, 224, 3)),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(train_gen.num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 3. Training
print("Training is starting. Please wait...")
model.fit(train_gen, epochs=5, validation_data=val_gen)

# 4. Save the actual FILE
model.save('plant_disease_model.h5')
print("SUCCESS: plant_disease_model.h5 has been created!")