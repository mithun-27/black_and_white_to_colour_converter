import streamlit as st
import numpy as np
import cv2
from PIL import Image
import os
from model import build_transfer_unet
from utils import preprocess_array, postprocess_image, color_pro_image
from tensorflow.keras.models import load_model

# Page configuration
st.set_page_config(page_title="AI Image Colorizer", page_icon="🎨", layout="wide")

# Custom CSS
st.markdown("""
<style>
    .stButton>button { width: 100%; border-radius: 5px; height: 3em; background-color: #ff4b4b; color: white; font-weight: bold; }
    .status-box { padding: 10px; border-radius: 5px; margin-bottom: 20px; }
</style>
""", unsafe_allow_html=True)

# Load DIY model with caching
@st.cache_resource
def load_diy_model(model_path="colorization_model.keras"):
    if os.path.exists(model_path):
        return load_model(model_path, compile=False)
    return build_transfer_unet()

# UI Layout
st.title("🎨 AI Image Colorization Dashboard")
st.markdown("Restore history with professional-grade AI or experiment with your own trained models.")

# Sidebar Settings
st.sidebar.title("🛠️ Model Configuration")
model_type = st.sidebar.radio("Select AI Engine:", ("Pro Model ", "DIY Model (Custom U-Net)"))

if model_type == "DIY Model (Custom U-Net)":
    diy_path = st.sidebar.text_input("Model Path", "colorization_model.keras")
    boost_factor = st.sidebar.slider("🎨 Color Intensity Boost", 0.0, 10.0, 1.0)
    st.sidebar.info("The DIY model uses your local training data. Use the boost factor if results are subtle.")
else:
    st.sidebar.success("🚀 Pro Model Active: Industry-standard colorization enabled.")
    st.sidebar.info("This model is pre-trained on millions of photos for realistic skin tones and environments.")

# Session State
if 'colored_result' not in st.session_state:
    st.session_state.colored_result = None

# Main Content
input_method = st.radio("Select Input Method:", ("Upload Image", "Capture Photo"))

img = None
if input_method == "Upload Image":
    uploaded_file = st.file_uploader("Choose a B&W image...", type=["jpg", "png", "jpeg"], label_visibility="collapsed")
    if uploaded_file is not None:
        img = Image.open(uploaded_file)
else:
    captured_file = st.camera_input("Take a photo")
    if captured_file is not None:
        img = Image.open(captured_file)

if img is not None:
    st.markdown("### Comparison")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Grayscale Input**")
        st.image(img, use_column_width=True)
        
    with col2:
        st.write(f"**AI Colorized ({model_type})**")
        
        if st.button("✨ Bring to Life!"):
            with st.spinner("AI is analyzing textures and intelligently applying colors..."):
                img_array = np.array(img.convert('RGB'))
                
                if model_type == "Pro Model (Pre-trained)":
                    result, error = color_pro_image(img_array)
                    if error:
                        st.error(error)
                    else:
                        st.session_state.colored_result = result
                else:
                    diy_model = load_diy_model(diy_path)
                    L, _ = preprocess_array(img_array)
                    L_input = np.expand_dims(L, axis=0)
                    predicted_AB = diy_model.predict(L_input)
                    boosted_AB = np.clip(predicted_AB[0] * boost_factor, -1, 1)
                    st.session_state.colored_result = postprocess_image(L, boosted_AB)
        
        if st.session_state.colored_result is not None:
            st.image(st.session_state.colored_result, use_column_width=True)
            
            # Download
            result_pil = Image.fromarray(st.session_state.colored_result)
            result_pil.save("colorized_pro_result.png")
            with open("colorized_pro_result.png", "rb") as file:
                st.download_button("Download Colorized Photo", file, "colorized.png", "image/png")

st.markdown("---")
st.markdown("<div style='text-align: center;'>coded by Mithun with ❤️</div>", unsafe_allow_html=True)
