from fastapi import FastAPI, UploadFile, File
import cv2
import numpy as np
import tensorflow as tf

app = FastAPI()
model = tf.keras.models.load_model('plant_disease_model.h5') #
class_names = ['Potato Early Blight', 'Potato Healthy', 'Potato Late Blight', 
               'Tomato Early Blight', 'Tomato Healthy', 'Tomato Late Blight']

@app.post("/predict")
async def predict_crop(file: UploadFile = File(...)):
    # 1. Read image from mobile request
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # 2. Run prediction on original image for accuracy
    img_prep = cv2.resize(img, (224, 224)) / 255.0
    predictions = model.predict(np.expand_dims(img_prep, axis=0))
    
    label = class_names[np.argmax(predictions)]
    confidence = float(np.max(predictions) * 100)

    # 3. Return diagnosis and Tamil advice
    return {
        "label": label,
        "confidence": f"{confidence:.2f}%",
        "tamil_advice": "நிவாரணம்: பாதிக்கப்பட்ட இலைகளை உடனே அகற்றவும்."
    }