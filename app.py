import streamlit as st
from ultralytics import YOLO
from PIL import Image
import tempfile
import os

st.set_page_config(
    page_title="Specialist Resistor Detector",
    page_icon="⚡",
    layout="centered"
)

st.title("⚡ Specialist Model: Common Resistor Value Detection")
st.write(
    "Upload or capture a resistor image. The YOLO Specialist model will detect "
    "the resistor and classify its common resistance value."
)

MODEL_PATH = "my_SP_1_Model.pt"

@st.cache_resource
def load_model():
    return YOLO(MODEL_PATH)

model = load_model()

st.sidebar.header("Prediction Settings")
confidence = st.sidebar.slider("Confidence Threshold", 0.05, 1.0, 0.25, 0.05)

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
    st.image(image, caption="Input Image", use_container_width=True)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
        image.save(temp_file.name)
        temp_path = temp_file.name

    results = model.predict(
        source=temp_path,
        conf=confidence,
        save=False
    )

    result = results[0]
    plotted_image = result.plot()

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

            st.success(f"Predicted Value: {class_name} Ω")
            st.write(f"Confidence: {conf_score:.2f}")

    os.remove(temp_path)
else:
    st.info("Upload an image or use the camera to start detection.")
