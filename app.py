import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image

# -------------------- PAGE CONFIG --------------------
st.set_page_config(page_title="Plant Disease AI", layout="centered")

# -------------------- BACKGROUND STYLE --------------------
def add_bg_and_style():
    st.markdown(
        """
        <style>
        .stApp {
            background: linear-gradient(rgba(0,0,0,0.6), rgba(0,0,0,0.6)),
                        url("https://images.unsplash.com/photo-1500382017468-9049fed747ef");
            background-size: cover;
            background-position: center;
        }
        h1, h2, h3, p {
            color: white !important;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.6);
        }
        [data-testid="stSidebar"] {
            background-color: rgba(0,0,0,0.85);
        }
        </style>
        """,
        unsafe_allow_html=True
    )

add_bg_and_style()

# -------------------- TITLE --------------------
st.title("🌿 Plant Disease Intelligence System")
st.markdown("### *AI-Powered Leaf Disease Detection (Tamil Supported)*")

# -------------------- LOAD MODEL --------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("plant_disease_model.h5")

model = load_model()

class_names = [
    "Potato Early Blight",
    "Potato Healthy",
    "Potato Late Blight",
    "Tomato Early Blight",
    "Tomato Healthy",
    "Tomato Late Blight"
]

tamil_labels = {
    "Potato Early Blight": "உருளைக்கிழங்கு – ஆரம்பகால கருகல் நோய்",
    "Potato Healthy": "உருளைக்கிழங்கு – ஆரோக்கியமானது",
    "Potato Late Blight": "உருளைக்கிழங்கு – பிந்தைய கருகல் நோய்",
    "Tomato Early Blight": "தக்காளி – ஆரம்பகால கருகல் நோய்",
    "Tomato Healthy": "தக்காளி – ஆரோக்கியமானது",
    "Tomato Late Blight": "தக்காளி – பிந்தைய கருகல் நோய்"
}

# -------------------- STRONG LEAF VALIDATION --------------------
def is_leaf_image(image):
    img = np.array(image)
    h, w, _ = img.shape

    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    # GREEN (LEAF)
    lower_green = np.array([30, 40, 40])
    upper_green = np.array([90, 255, 255])
    green_mask = cv2.inRange(hsv, lower_green, upper_green)
    green_ratio = np.sum(green_mask > 0) / green_mask.size

    # SKIN (FACE REJECTION)
    lower_skin = np.array([0, 20, 70])
    upper_skin = np.array([20, 255, 255])
    skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)
    skin_ratio = np.sum(skin_mask > 0) / skin_mask.size

    # LEAF SHAPE CHECK
    contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return False

    largest_area = max(cv2.contourArea(c) for c in contours)
    image_area = h * w

    if green_ratio < 0.12:
        return False
    if skin_ratio > 0.08:
        return False
    if largest_area < 0.15 * image_area:
        return False

    return True

# -------------------- SIDEBAR --------------------
st.sidebar.header("⚙️ Settings")
confidence_threshold = st.sidebar.slider("Confidence Threshold (%)", 0, 100, 80)

# -------------------- INPUT METHOD --------------------
option = st.radio(
    "Choose Input Method",
    ("Live Camera", "Upload Image"),
    horizontal=True
)

if option == "Live Camera":
    st.info("📸 இலைக்கு மட்டும் கேமராவை வைத்து படம் எடுக்கவும் (முகம் / அறை வேண்டாம்)")
    img_file = st.camera_input("Scan Leaf")
else:
    img_file = st.file_uploader("Upload Leaf Image", type=["jpg", "jpeg", "png"])

# -------------------- MAIN PIPELINE --------------------
if img_file is not None:
    image = Image.open(img_file).convert("RGB")
    st.image(image, caption="Target Leaf Preview", use_container_width=True)

    # ❌ FACE / NON-LEAF BLOCK
    if not is_leaf_image(image):
        st.error("❌ இது தாவர இலை படம் அல்ல. தயவுசெய்து உண்மையான இலை படத்தை மட்டும் பயன்படுத்தவும்.")
        st.stop()

    # PREPROCESS
    img_array = np.array(image.resize((224, 224))) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # PREDICTION
    with st.spinner("🌱 இலை ஆய்வு நடைபெறுகிறது..."):
        preds = model.predict(img_array, verbose=0)
        idx = np.argmax(preds)
        label = class_names[idx]
        confidence = np.max(preds) * 100

    st.divider()

    # RESULTS
    if confidence < confidence_threshold:
        st.warning("⚠️ துல்லியம் குறைவு. தெளிவான இலை படத்தை மீண்டும் எடுக்கவும்.")
    else:
        is_healthy = "Healthy" in label
        tamil_name = tamil_labels[label]

        if is_healthy:
            st.success("✅ ஆய்வு முடிவு : ஆரோக்கியமான செடி")
            st.balloons()
        else:
            st.error("🚨 ஆய்வு முடிவு : நோய் கண்டறியப்பட்டுள்ளது")

        col1, col2 = st.columns(2)
        with col1:
            st.info(f"🌿 **கண்டறியப்பட்ட நிலை:**\n\n{tamil_name}")
        with col2:
            st.metric("துல்லியம் (Confidence)", f"{confidence:.2f} %")
