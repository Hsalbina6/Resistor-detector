import os
# Force uninstall the broken OpenCV version on Streamlit Cloud to prevent crashing
os.system("pip uninstall -y opencv-python")

import streamlit as st
from ultralytics import YOLO
from PIL import Image
import tempfile

# Helper function to format raw numbers into readable Ohms/Kilo-ohms
def format_resistance(value):
    try:
        val = int(value)
        if val >= 1000:
            return f"{val/1000:g} kΩ"
        return f"{val} Ω"
    except ValueError:
        return f"{value} Ω"

# 1. Page Config
st.set_page_config(
    page_title="Resistor Value Detector",
    page_icon="⚡",
    layout="wide" 
)

# ==========================================
# SIDEBAR CONFIGURATION
# ==========================================
st.sidebar.title("⚙️ Model Settings")

# Model Selection
model_choice = st.sidebar.radio(
    "Select AI Logic:",
    ["Specialist Model", "Generalist Model (Coming Soon)", "Smart Logic (Coming Soon)"]
)

st.sidebar.markdown("---")

# Notes Section
st.sidebar.markdown("### 📝 Model Notes")
st.sidebar.info(
    "**Specialist:** Limited to the list below, but highly robust. Can detect values even in tough conditions (blur, angles, poor lighting).\n\n"
    "**Generalist:** Not limited to a list. Reads the exact color bands, but requires good image conditions to distinguish colors accurately.\n\n"
    "**Smart Logic:** Combines both for the absolute best outcome."
)

st.sidebar.markdown("---")

# Supported Values List (Exact match to the 10 trained YOLO classes)
st.sidebar.markdown("### 🎯 Specialist Values")
st.sidebar.markdown("""
* **Ohms (Ω):** 10, 220, 330
* **Kilo-ohms (kΩ):** 1k, 4.7k, 6.8k, 8.2k, 9.2k, 10k, 20k
""")


# ==========================================
# MAIN CONTENT AREA
# ==========================================

# --- SPECIALIST MODEL ---
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
        st.image(image, caption="Input Image", width=400)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
            image.save(temp_file.name)
            temp_path = temp_file.name

        results = model.predict(
            source=temp_path,
            conf=confidence,
            save=False
        )

        result = results[0]
        # [..., ::-1] converts YOLO's BGR output to Streamlit's required RGB format
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

# --- GENERALIST MODEL (COMING SOON) ---
elif model_choice == "Generalist Model (Coming Soon)":
    st.title("📊 Generalist Model (Color Band Reader)")
    st.write(
        "This model processes cropped resistors to explicitly read and decode their color bands. "
        "Unlike the Specialist model, it is not limited to a pre-defined list of values and can decode *any* standard resistor. "
        "However, it requires clear lighting and good conditions to accurately distinguish colors."
    )
    st.warning("🚧 This feature is currently under development. Check back soon!")

# --- SMART LOGIC (COMING SOON) ---
elif model_choice == "Smart Logic (Coming Soon)":
    st.title("🧠 Smart Logic: Hybrid Detection")
    st.write(
        "The Smart Logic system serves as the ultimate brain of the application. It strategically leverages both the Specialist and Generalist models to ensure the highest possible accuracy for every detection."
    )
    
    st.markdown("### 🔄 Smart Logic Workflow")
    
    st.info("**1. Input Image** ➔ Passes through Base Resistor Detection Model ➔ Extracts **Cropped Resistors**")
    
    col1, col2 = st.columns(2)
    with col1:
        st.success("**2. Specialist Analysis**\nThe cropped resistor is first fed to the Specialist Model.")
    with col2:
        st.warning("**3. Confidence Check**\nDoes the Specialist Model recognize it with high confidence?")
        
    st.markdown("""
    * ✅ **Yes (High Confidence):** Directly output the Specialist predicted value.
    * ❌ **No (Low Confidence / Not on list):** Route the cropped image to the **Generalist Model** to read the exact color bands, then output the value.
    """)
    
    st.warning("🚧 This logic pipeline is currently under development. Check back soon!")
