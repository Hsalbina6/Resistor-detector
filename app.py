import os
# Force uninstall the broken OpenCV version on Streamlit Cloud to prevent crashing
os.system("pip uninstall -y opencv-python")

import streamlit as st
from ultralytics import YOLO
from PIL import Image, ImageDraw
import tempfile
import numpy as np


# ==========================================
# HELPER FUNCTIONS
# ==========================================

def format_resistance(value):
    try:
        val = int(value)
        if val >= 1000:
            return f"{val/1000:g} kΩ"
        return f"{val} Ω"
    except ValueError:
        return f"{value} Ω"


def analyze_image_quality(image):
    img_gray = np.array(image.convert("L"))

    brightness = np.mean(img_gray)
    blur_score = np.var(img_gray)

    feedback = []

    if brightness < 70:
        feedback.append("Image may be too dark. Try better lighting.")
    elif brightness > 210:
        feedback.append("Image may be too bright. Reduce glare or strong light.")

    if blur_score < 500:
        feedback.append("Image may be blurry. Try holding the camera steady.")

    if not feedback:
        feedback.append("Image quality looks acceptable.")

    return brightness, blur_score, feedback


def add_capture_guide(image):
    guided_image = image.copy()
    draw = ImageDraw.Draw(guided_image)

    w, h = guided_image.size

    box_w = int(w * 0.75)
    box_h = int(h * 0.35)

    left = int((w - box_w) / 2)
    top = int((h - box_h) / 2)
    right = left + box_w
    bottom = top + box_h

    draw.rectangle(
        [left, top, right, bottom],
        outline="red",
        width=6
    )

    return guided_image


# ==========================================
# PAGE CONFIG
# ==========================================

st.set_page_config(
    page_title="Resistor Value Detector",
    page_icon="⚡",
    layout="wide"
)


# ==========================================
# SIDEBAR CONFIGURATION
# ==========================================

st.sidebar.title("⚙️ Model Settings")

model_choice = st.sidebar.radio(
    "Select AI Logic:",
    ["Specialist Model", "Generalist Model (Coming Soon)", "Smart Logic (Coming Soon)"]
)

st.sidebar.markdown("---")

st.sidebar.markdown("### 📝 Model Notes")
st.sidebar.info(
    "**Specialist:** Limited to the list below, but highly robust. Can detect values even in tough conditions "
    "(blur, angles, poor lighting).\n\n"
    "**Generalist:** Not limited to a list. Reads the exact color bands, but requires good image conditions "
    "to distinguish colors accurately.\n\n"
    "**Smart Logic:** Combines both for the absolute best outcome."
)

st.sidebar.markdown("---")

st.sidebar.markdown("### 🎯 Specialist Values")
st.sidebar.markdown("""
* **Ohms (Ω):** 10, 220, 330
* **Kilo-ohms (kΩ):** 1k, 4.7k, 6.8k, 8.2k, 9.2k, 10k, 20k
""")


# ==========================================
# SPECIALIST MODEL
# ==========================================

if model_choice == "Specialist Model":
    st.title("⚡ Specialist Model: Common Resistor Detection")
    st.write(
        "Upload or capture a resistor image. The YOLO Specialist model will detect "
        "the resistor and classify its common resistance value."
    )

    MODEL_PATH = "my_SP_1_Model.pt"

    @st.cache_resource
    def load_model():
        return YOLO(MODEL_PATH)

    model = load_model()

    st.markdown("### Prediction Settings")
    confidence = st.slider("Confidence Threshold", 0.05, 1.0, 0.25, 0.05)

    option = st.radio(
        "Choose input method:",
        ["Upload Image", "Use Camera"]
    )

    image_file = None

    if option == "Upload Image":
        image_file = st.file_uploader(
            "Upload resistor image",
            type=["jpg", "jpeg", "png"]
        )
    else:
        image_file = st.camera_input("Take a resistor picture")

    if image_file is not None:
        image = Image.open(image_file).convert("RGB")

        guide_col, quality_col = st.columns([1.2, 1])

        with guide_col:
            st.subheader("Capture Guide")
            guided_image = add_capture_guide(image)
            st.image(
                guided_image,
                caption="Place the resistor inside the red guide box for best results.",
                width=450
            )

        with quality_col:
            st.subheader("Image Quality Feedback")
            brightness, blur_score, feedback = analyze_image_quality(image)

            st.write(f"**Brightness Score:** {brightness:.1f}")
            st.write(f"**Blur Score:** {blur_score:.1f}")

            for message in feedback:
                if message == "Image quality looks acceptable.":
                    st.success(message)
                else:
                    st.warning(message)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
            image.save(temp_file.name)
            temp_path = temp_file.name

        results = model.predict(
            source=temp_path,
            conf=confidence,
            save=False
        )

        result = results[0]
        plotted_image = result.plot()[..., ::-1]

        st.subheader("Prediction Result")
        st.image(plotted_image, caption="Detected Resistor Value", use_container_width=True)

        st.subheader("Detected Classes")

        if len(result.boxes) == 0:
            st.warning("No resistor value detected. Try a clearer image or lower the confidence threshold.")
        else:
            for box in result.boxes:
                class_id = int(box.cls[0])
                class_name = model.names[class_id]
                conf_score = float(box.conf[0])

                st.success(f"Predicted Value: **{format_resistance(class_name)}**")
                st.write(f"Raw Model Class: {class_name}")
                st.write(f"Confidence: {conf_score:.2f}")

        os.remove(temp_path)

    else:
        st.info("Upload an image or use the camera to start detection.")


# ==========================================
# GENERALIST MODEL
# ==========================================

elif model_choice == "Generalist Model (Coming Soon)":
    st.title("🧠 Generalist Model (Color Band Reader)")
    st.write(
        "This model processes cropped resistors to explicitly read and decode their color bands. "
        "Unlike the Specialist model, it is not limited to a pre-defined list of values and can decode any standard resistor. "
        "However, it requires clear lighting and good conditions to accurately distinguish colors."
    )
    st.warning("🚧 This feature is currently under development. Check back soon!")


# ==========================================
# SMART LOGIC
# ==========================================

elif model_choice == "Smart Logic (Coming Soon)":
    st.title("🧠 Smart Logic: Hybrid Detection")
    st.write(
        "The Smart Logic system serves as the decision layer of the application. "
        "It strategically combines the Specialist and Generalist models to improve reliability."
    )

    st.markdown("### 🔄 Smart Logic Workflow")

    st.info("**1. Input Image** ➔ Base resistor detection ➔ Extract cropped resistor")

    col1, col2 = st.columns(2)

    with col1:
        st.success(
            "**2. Specialist Analysis**\n\n"
            "The cropped resistor is checked against the known Specialist value database."
        )

    with col2:
        st.warning(
            "**3. Generalist Analysis**\n\n"
            "The cropped resistor is decoded by reading individual color bands."
        )

    st.markdown("""
    ### Decision Logic

    ✅ **If both models agree:**  
    Display the value with **High Confidence**.

    ⚠️ **If they disagree but the value exists in the Specialist list:**  
    Display the Specialist result with **Medium Confidence**.

    ❌ **If the value does not exist in the Specialist list:**  
    Use the Generalist result with **Low Confidence**.
    """)

    st.warning("🚧 This logic pipeline is currently under development. Check back soon!")
