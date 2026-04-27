# ⚡ Smart Resistor Detection System

A Streamlit-based application that uses a trained YOLO model to detect and classify resistor values from images.

This project demonstrates a **Specialist AI model**, with planned expansion into a **Generalist model** and a **Smart Logic system** for improved reliability.

---

## 🚀 Features

### ✅ Specialist Model (Active)
- Detects **common resistor values directly from images**
- Works well under **challenging conditions** (blur, angles, lighting)
- Limited to a predefined set of values

### 🔍 Image Diagnostics
- Brightness analysis
- Blur detection (sharpness)
- Real-time feedback to improve capture quality
- Visual **capture guide overlay**

### 📦 Session Inventory
- Save detected resistor values during session
- View results in a structured table
- Export inventory as a **CSV file**
- Clear inventory anytime

---

## 🧠 Model Architecture

### Specialist Model
- Directly classifies resistor values from full images
- Robust to real-world conditions
- Limited to known values

### Generalist Model (Coming Soon)
- Reads **color bands** instead of fixed values
- Can decode any resistor
- Requires **high-quality images**

### Smart Logic System (Coming Soon)
- Combines Specialist + Generalist
- Decision-based output:
  - High confidence → models agree
  - Medium → fallback to Specialist
  - Low → fallback to Generalist

---

## 🎯 Supported Resistor Values (Specialist)

- **Ohms (Ω):** 10, 220, 330  
- **Kilo-ohms (kΩ):** 1k, 4.7k, 6.8k, 8.2k, 9.2k, 10k, 20k  

---

## 🧪 Sample Images

The `EXAMPLES/` folder contains **unseen test images** that were not used during training.  
These are provided for demonstration and evaluation purposes.

---

## 🛠️ How to Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
