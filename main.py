import os
os.system("chcp 65001")  # Force UTF-8 for Windows (Tamil support)

import cv2
import numpy as np
import tensorflow as tf
from segmentation import apply_borb_segmentation
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

# -------------------- CONSOLE --------------------
console = Console()

# -------------------- TAMIL PREVENTIONS --------------------
preventions = {
    "Potato Early Blight": "நிவாரணம்: மேன்கோசெப் 2 கிராம்/லிட்டர் வீதம் தெளிக்கவும்.",
    "Potato Late Blight": "நிவாரணம்: பாதிக்கப்பட்ட செடிகளை உடனே அகற்றவும்.",
    "Tomato Early Blight": "நிவாரணம்: வேப்ப எண்ணெய் தெளிக்கவும். காற்றோட்டம் அவசியம்.",
    "Tomato Late Blight": "நிவாரணம்: போர்டோ கலவை அல்லது பூஞ்சாணக் கொல்லிகளைப் பயன்படுத்தவும்.",
    "Potato Healthy": "பயிர் ஆரோக்கியமானது. வழக்கமான பராமரிப்பு போதுமானது.",
    "Tomato Healthy": "பயிர் ஆரோக்கியமானது. தொடர்ந்த கண்காணிப்பு பரிந்துரை செய்யப்படுகிறது."
}

# -------------------- LOAD MODEL --------------------
try:
    model = tf.keras.models.load_model("plant_disease_model.h5")
    console.print("[bold green]✔ Model loaded successfully![/bold green]")
except Exception as e:
    console.print(f"[bold red]✘ Model loading failed: {e}[/bold red]")
    exit()

# -------------------- CLASS LABELS --------------------
class_names = [
    "Potato Early Blight",
    "Potato Healthy",
    "Potato Late Blight",
    "Tomato Early Blight",
    "Tomato Healthy",
    "Tomato Late Blight"
]

# -------------------- DISPLAY FUNCTION --------------------
def display_terminal_results(label, confidence):
    is_healthy = "Healthy" in label
    status_ta = "ஆரோக்கியமானது" if is_healthy else "பாதிக்கப்பட்டுள்ளது"
    border_color = "green" if is_healthy else "red"

    instruction = preventions.get(label, "தகவல் இல்லை")

    results_text = f"""
Diagnosis       : {label}
Confidence      : {confidence:.2f} %

Status          : {status_ta}

விவசாயிகளுக்கான அறிவுரை:
{instruction}
"""

    console.print(
        Panel.fit(
            Text(results_text, justify="left"),
            title="🌱 Tamil Nadu Crop Health AI",
            border_style=border_color,
            padding=(1, 2)
        )
    )

# -------------------- MAIN PREDICTION PIPELINE --------------------
def run_full_prediction(img_path):
    original_img = cv2.imread(img_path)

    if original_img is None:
        console.print("[bold red]✘ Image not found! Check path.[/bold red]")
        return

    # ---- BorB Segmentation (Visualization only) ----
    _, _, segmented_img = apply_borb_segmentation(original_img)

    # ---- Classification (Use original image) ----
    img = cv2.resize(original_img, (224, 224))
    img = np.expand_dims(img, axis=0) / 255.0

    predictions = model.predict(img, verbose=0)
    label = class_names[np.argmax(predictions)]
    confidence = np.max(predictions) * 100

    # ---- Terminal Output ----
    display_terminal_results(label, confidence)

    # ---- Visual Output ----
    cv2.imshow("AI Diagnosis - Original Image", original_img)
    cv2.imshow("BorB Segmentation View", segmented_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# -------------------- RUN --------------------
if __name__ == "__main__":
    test_image_path = "dataset/valid/tomato_late/tomato_late_4.JPG"
    run_full_prediction(test_image_path)
