import tensorflow as tf

# Step 1: Load your Keras model (.h5)
model = tf.keras.models.load_model("plant_disease_model.h5")
print("Model loaded successfully!")

# Step 2: Create TFLite converter
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# Step 3: Optional: Enable quantization to reduce file size
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# Step 4: Convert the model to TFLite format
tflite_model = converter.convert()

# Step 5: Save the converted model
with open("plant_disease_model.tflite", "wb") as f:
    f.write(tflite_model)

print("Model converted and saved as plant_disease_model.tflite")
