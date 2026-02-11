import tensorflow as tf

# Load your large .h5 model
model = tf.keras.models.load_model("plant_disease_model.h5")
print("Model loaded successfully!")
