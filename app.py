import os
# Force uninstall the broken OpenCV version on Streamlit Cloud to prevent crashing
os.system("pip uninstall -y opencv-python")

import streamlit as st
import pandas as pd
from ultralytics import YOLO
from PIL import Image, ImageDraw
import tempfile
import numpy as np


# ==========================================
# SESSION STATE INITIALIZATION
# ==========================================
if "inventory" not in st.session_state:
    st.session_state.inventory = []


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
    status = "good"

    if brightness < 70:
        feedback.append("Image may be too dark. Try better lighting.")
        status = "warning"
    elif brightness > 210:
        feedback.append("Image may be too bright. Reduce glare or strong light.")
        status = "warning"

    if blur_score < 500:
        feedback.append("Image may be blurry. Try holding the camera steady.")
        status = "warning"

    if not feedback:
        feedback.append("Image quality looks acceptable.")

    return brightness, blur_score, feedback, status


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
        outline="#FF4B4B", 
        width=5
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

st.sidebar.title("⚙️ Navigation")

model_choice = st.sidebar.radio(
    "Select AI Logic:",
    ["Specialist Model", "Generalist Model (Coming Soon)", "Smart Logic (Coming Soon)"]
)

st.sidebar.markdown("---")

# Inventory Tracker UI
st.sidebar.markdown("### 📦 Session Inventory")
if st.session_state.inventory:
    # Convert session state list to pandas DataFrame for clean display
    df_inventory = pd.DataFrame(st.session_state.inventory)
    
    # Display table
    st.sidebar.dataframe(df_inventory, hide_index=True, use_container_width=True)
    
    # CSV Download Button
    csv = df_inventory.to_csv(index=False).encode('utf-8')
    st.sidebar.download_button(
        label="📥 Download CSV",
        data=csv,
        file_name='resistor_inventory.csv',
        mime='text/csv',
    )
    
    # Clear Inventory Button
    if st.sidebar.button("🗑️ Clear Inventory"):
        st.session_state.inventory = []
        st.rerun()
else:
    st.sidebar.info("Inventory is empty. Scan components and save them to build your list.")

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
        "the resistor and classify its value."
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
        ["Upload Image", "Use Camera"],
        horizontal=True
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
        
        st.markdown("### Analysis")
        
        # --- IMAGE DIAGNOSTICS SECTION ---
        with st.expander("🔍 View Image Quality Diagnostics", expanded=False):
            brightness, blur_score, feedback, status = analyze_image_quality(image)
            
            diag_col1, diag_col2 = st.columns([1, 1])
            
            with diag_col1:
                st.markdown("#### Capture Guide")
                guided_image = add_capture_guide(image)
                st.image(guided_image, caption="Ideal framing area", use_container_width=True)
                
            with diag_col2:
                st.markdown("#### Quality Metrics")
                
                met_col1, met_col2 = st.columns(2)
                met_col1.metric("Brightness", f"{brightness:.0f} / 255")
                met_col2.metric("Sharpness", f"{blur_score:.0f}")
                
                st.markdown("#### Feedback")
                for message in feedback:
                    if message == "Image quality looks acceptable.":
                        st.success(message)
                    else:
                        st.warning(message)

        # --- YOLO PREDICTION SECTION ---
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
            image.save(temp_file.name)
            temp_path = temp_file.name

        with st.spinner("Analyzing image..."):
            results = model.predict(
                source=temp_path,
                conf=confidence,
                save=False
            )

        result = results[0]
        plotted_image = result.plot()[..., ::-1]

        # Final Result Display
        res_col1, res_col2 = st.columns([1.5, 1])
        
        with res_col1:
            st.image(plotted_image, caption="AI Detection Result", use_container_width=True)
            
        with res_col2:
            st.subheader("Detected Values")
            
            if len(result.boxes) == 0:
                st.error("No resistor detected.")
                st.info("Try adjusting the confidence threshold or taking a clearer picture.")
            else:
                detected_items = [] # Temporary list to hold current predictions
                
                for box in result.boxes:
                    class_id = int(box.cls[0])
                    class_name = model.names[class_id]
                    conf_score = float(box.conf[0])
                    formatted_val = format_resistance(class_name)

                    # Display Prediction
                    st.success(f"**{formatted_val}**")
                    st.progress(conf_score, text=f"Confidence: {conf_score*100:.1f}%")
                    
                    # Add to temporary array in case user wants to save
                    detected_items.append({
                        "Value": formatted_val,
                        "Confidence": f"{conf_score*100:.1f}%"
                    })
                
                # Inventory Saving Action
                st.markdown("---")
                if st.button("💾 Save to Inventory"):
                    for item in detected_items:
                        st.session_state.inventory.append(item)
                    st.success(f"Successfully added {len(detected_items)} item(s) to your session inventory!")

        os.remove(temp_path)

    else:
        st.info("Upload an image or use the camera to start detection.")


# ==========================================
# GENERALIST MODEL
# ==========================================

elif model_choice == "Generalist Model (Coming Soon)":
    st.title("📊 Generalist Model (Color Band Reader)")
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

    ✅ **If both models agree:** Display the value with **High Confidence**.

    ⚠️ **If they disagree but the value exists in the Specialist list:** Display the Specialist result with **Medium Confidence**.

    ❌ **If the value does not exist in the Specialist list:** Use the Generalist result with **Low Confidence**.
    """)

    st.warning("🚧 This logic pipeline is currently under development. Check back soon!")
