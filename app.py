# ===============================
# 🌿 LeafLens AI - Final App
# ===============================

import gradio as gr
from PIL import Image
import numpy as np
import tensorflow as tf

# ===============================
# LOAD MODEL (YOUR BACKEND)
# ===============================
model = tf.keras.models.load_model("model.h5")

class_names = [
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Septoria_leaf_spot"
]

# ===============================
# DISEASE INFO (FRONTEND DATA)
# ===============================
DISEASES = {
    "Tomato — Early Blight": {
        "severity": "MODERATE",
        "desc": "Common fungal disease with circular brown spots.",
        "treatment": "Use fungicide, remove infected leaves."
    },
    "Tomato — Late Blight": {
        "severity": "CRITICAL",
        "desc": "Serious disease spreading rapidly in wet conditions.",
        "treatment": "Apply fungicide immediately and isolate plants."
    },
    "Tomato — Septoria Leaf Spot": {
        "severity": "MODERATE",
        "desc": "Small circular spots with dark borders.",
        "treatment": "Remove infected leaves and apply fungicide."
    }
}

# ===============================
# MODEL PREDICTION FUNCTION
# ===============================
def run_model(pil_image):
    img = pil_image.convert("RGB")
    img = img.resize((224,224))
    img = np.array(img).astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)

    preds = model.predict(img, verbose=0)

    class_index = np.argmax(preds[0])
    confidence = float(np.max(preds[0]))

    print("Raw Predictions:", preds)

    # confidence filter
    if confidence < 0.7:
        return "Uncertain (Try clearer image)", confidence

    label = class_names[class_index]

    # mapping to UI label
    mapping = {
        "Tomato___Early_blight": "Tomato — Early Blight",
        "Tomato___Late_blight": "Tomato — Late Blight",
        "Tomato___Septoria_leaf_spot": "Tomato — Septoria Leaf Spot"
    }

    return mapping[label], confidence
# ===============================
# MAIN ANALYZE FUNCTION
# ===============================
def analyze(image):
    try:
        if image is None:
            return "Upload image first"

        pil = Image.fromarray(image).convert("RGB")
        label, confidence = run_model(pil)

        info = DISEASES.get(label)

        # handle unknown label safely
        if info is None:
            return f"""
            ⚠ Prediction: {label}
            📊 Confidence: {confidence*100:.2f}%
            """

        result = f"""
        🌿 Disease: {label}

        📊 Confidence: {confidence*100:.2f}%

        ⚠ Severity: {info['severity']}

        🧾 Description:
        {info['desc']}

        💊 Treatment:
        {info['treatment']}
        """

        return result

    except Exception as e:
        return f"Error: {str(e)}"

# ===============================
# GRADIO UI (FRONTEND)
# ===============================
with gr.Blocks() as demo:
    gr.Markdown("# 🌿 AI Plant Disease Detection")

    image_input = gr.Image(type="numpy", label="Upload Leaf Image")
    output = gr.Textbox(label="Result")

    btn = gr.Button("Analyze")

    btn.click(fn=analyze, inputs=image_input, outputs=output)

# ===============================
# LAUNCH
# ===============================
if __name__ == "__main__":
    demo.launch(debug=True)