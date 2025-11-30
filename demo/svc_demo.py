import streamlit as st
import pandas as pd
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input
import joblib
import os
import time
import plotly.graph_objects as go

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Phân loại Động vật biển", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    [data-testid="stSidebar"] {
        background-color: #1a1a2e;
    }
    
    .main {
        background-color: #0f0f1e;
        color: #e0e0e0;
    }
    
    h1 {
        color: #00d4ff;
        text-align: center;
        margin-bottom: 30px;
    }
    
    h2 {
        color: #00d4ff;
        margin-top: 20px;
    }
    
    .stDataFrame {
        margin-top: 20px;
    }
    
    .stInfo, .stSuccess, .stError, .stWarning {
        margin: 10px 0;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("Phân loại Động vật biển")

# ============================================================================
# CONSTANTS
# ============================================================================

CLASSES = ['Clams', 'Corals', 'Crabs', 'Dolphin', 'Eel', 'Fish', 'Jelly Fish', 'Lobster', 'Nudibranchs', 'Octopus', 'Otter', 'Penguin', 'Puffers', 'Sea Rays', 'Sea Urchins', 'Seahorse', 'Seal', 'Sharks', 'Shrimp', 'Squid', 'Starfish', 'Turtle_Tortoise', 'Whale']

CLASS_INFO = {
    'Clams': {'Scientific Name': 'Bivalvia', 'Size': '1 cm - 1.2 m', 'Habitat': 'Đáy cát hoặc bùn'},
    'Corals': {'Scientific Name': 'Anthozoa', 'Size': 'Đa dạng (tập đoàn lớn)', 'Habitat': 'Biển nhiệt đới, rạn san hô'},
    'Crabs': {'Scientific Name': 'Brachyura', 'Size': 'Vài mm - 4 m (sải chân)', 'Habitat': 'Đại dương, nước ngọt, trên cạn'},
    'Dolphin': {'Scientific Name': 'Delphinidae', 'Size': '1.7 m - 9.5 m', 'Habitat': 'Đại dương, một số sông'},
    'Eel': {'Scientific Name': 'Anguilliformes', 'Size': '5 cm - 4 m', 'Habitat': 'Nước ngọt, nước mặn'},
    'Fish': {'Scientific Name': 'Pisces', 'Size': 'Rất đa dạng', 'Habitat': 'Nước ngọt, nước mặn'},
    'Jelly Fish': {'Scientific Name': 'Medusozoa', 'Size': '1 mm - 2 m (đường kính)', 'Habitat': 'Tất cả đại dương, một số nước ngọt'},
    'Lobster': {'Scientific Name': 'Nephropidae', 'Size': '25 cm - 60 cm', 'Habitat': 'Đáy đá, cát hoặc bùn'},
    'Nudibranchs': {'Scientific Name': 'Nudibranchia', 'Size': '4 mm - 60 cm', 'Habitat': 'Đại dương, chủ yếu ở đáy'},
    'Octopus': {'Scientific Name': 'Octopoda', 'Size': '1 cm - 9 m', 'Habitat': 'Rạn san hô, nước nổi, đáy biển'},
    'Otter': {'Scientific Name': 'Lutrinae', 'Size': '0.6 m - 1.8 m', 'Habitat': 'Bờ biển, sông'},
    'Penguin': {'Scientific Name': 'Spheniscidae', 'Size': '30 cm - 1.1 m', 'Habitat': 'Nam bán cầu, ven biển'},
    'Puffers': {'Scientific Name': 'Tetraodontidae', 'Size': '2.5 cm - 1 m', 'Habitat': 'Biển nhiệt đới và cận nhiệt đới'},
    'Sea Rays': {'Scientific Name': 'Batoidea', 'Size': 'Vài cm - 7 m (chiều rộng)', 'Habitat': 'Vùng nước ven bờ, biển sâu'},
    'Sea Urchins': {'Scientific Name': 'Echinoidea', 'Size': '3 cm - 10 cm', 'Habitat': 'Đáy biển, rạn san hô'},
    'Seahorse': {'Scientific Name': 'Hippocampus', 'Size': '1.5 cm - 35 cm', 'Habitat': 'Thảm cỏ biển, rạn san hô'},
    'Seal': {'Scientific Name': 'Pinnipedia', 'Size': '1 m - 5 m', 'Habitat': 'Vùng nước cực và ôn đới'},
    'Sharks': {'Scientific Name': 'Selachimorpha', 'Size': '17 cm - 12 m', 'Habitat': 'Tất cả các đại dương'},
    'Shrimp': {'Scientific Name': 'Caridea', 'Size': 'Vài mm - 20 cm', 'Habitat': 'Nước mặn và nước ngọt'},
    'Squid': {'Scientific Name': 'Teuthida', 'Size': 'Vài cm - 13 m', 'Habitat': 'Đại dương mở, biển sâu'},
    'Starfish': {'Scientific Name': 'Asteroidea', 'Size': '12 cm - 24 cm', 'Habitat': 'Đáy biển, rạn san hô'},
    'Turtle_Tortoise': {'Scientific Name': 'Testudines', 'Size': '10 cm - 2 m', 'Habitat': 'Đại dương (rùa), đất liền (rùa cạn)'},
    'Whale': {'Scientific Name': 'Cetacea', 'Size': '2.6 m - 29.9 m', 'Habitat': 'Tất cả các đại dương'}
}

IMG_SIZE = 224

# ============================================================================
# MODEL LOADING
# ============================================================================

@st.cache_resource
def load_models():
    """Load pretrained models and scaler"""
    try:
        # Load Feature Extractor (ResNet50)
        feature_extractor = load_model('resnet50_feature_extractor.h5')
        
        # Load Scaler
        scaler = joblib.load('scaler_resnet_feature_extraction.pkl')
            
        # Load SVC Model
        svc_model = joblib.load('svc_model_tuned.pkl')
            
        return feature_extractor, scaler, svc_model
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None, None

def preprocess_image(image_pil):
    """Preprocess image for ResNet50 inference"""
    # Convert to RGB if necessary
    if image_pil.mode != 'RGB':
        image_pil = image_pil.convert('RGB')
    
    # Resize to IMG_SIZE
    image_pil = image_pil.resize((IMG_SIZE, IMG_SIZE), Image.Resampling.LANCZOS)
    
    # Convert to numpy array
    image_array = np.array(image_pil)
    
    # Expand dims to create batch (1, 224, 224, 3)
    image_array = np.expand_dims(image_array, axis=0)
    
    # Preprocess input (ResNet50 specific preprocessing)
    preprocessed_image = preprocess_input(image_array)
    
    return preprocessed_image

def predict_image(feature_extractor, scaler, svc_model, image_tensor):
    """Get predictions from model pipeline"""
    # 1. Extract features
    features = feature_extractor.predict(image_tensor, verbose=0)
    
    # Flatten if necessary (if output is 4D like (1, 7, 7, 2048))
    if len(features.shape) > 2:
        features = features.reshape(features.shape[0], -1)
        
    # 2. Scale features
    scaled_features = scaler.transform(features)
    
    # 3. Predict with SVC
    # Check if probability is available
    try:
        probabilities = svc_model.predict_proba(scaled_features)
        confidence = np.max(probabilities) * 100
        predicted_idx = np.argmax(probabilities)
    except Exception:
        # Fallback if no probability (SVC needs probability=True)
        predicted_class_label = svc_model.predict(scaled_features)[0]
        # Find index of class label
        try:
            # If the model returns a string label
            if isinstance(predicted_class_label, str):
                predicted_idx = CLASSES.index(predicted_class_label)
            else:
                # If the model returns an integer index
                predicted_idx = int(predicted_class_label)
        except:
             # Just return 0 and 100% if we can't map
             predicted_idx = 0 
             
        confidence = 100.0 # Placeholder if no proba
        
    return predicted_idx, confidence

# ============================================================================
# SIDEBAR - MODEL STATUS
# ============================================================================

with st.sidebar:
    st.subheader("Model Status")
    
    feature_extractor, scaler, svc_model = load_models()
    
    if feature_extractor is not None:
        st.success("✓ Đã tải Feature Extractor (ResNet50)")
    else:
        st.error("✗ Lỗi Feature Extractor")
        
    if scaler is not None:
        st.success("✓ Đã tải Scaler")
    else:
        st.error("✗ Lỗi Scaler")
        
    if svc_model is not None:
        st.success("✓ Đã tải SVC Model")
    else:
        st.error("✗ Lỗi SVC Model")

# ============================================================================
# MAIN CONTENT
# ============================================================================

col_left, col_right = st.columns([1, 2], gap="large")

with col_left:
    st.subheader("Tải ảnh lên")
    uploaded_file = st.file_uploader(
        "Chọn một file ảnh",
        type=["jpg", "jpeg", "png", "webp"],
        help="Tối đa 200MB • JPG, JPEG, PNG, WEBP"
    )
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, use_container_width=True, caption="Ảnh đã tải lên")

with col_right:
    if uploaded_file is not None and feature_extractor is not None and scaler is not None and svc_model is not None:
        st.subheader("Đang xử lý...")
        
        # Process image
        start_time = time.time()
        
        # Preprocess
        image_tensor = preprocess_image(image)
        
        # Predict
        predicted_idx, confidence = predict_image(feature_extractor, scaler, svc_model, image_tensor)
        
        inference_time = time.time() - start_time
        
        predicted_class = CLASSES[predicted_idx] if predicted_idx < len(CLASSES) else f"Class {predicted_idx}"
        
        # Display results
        st.success("✓ Dự đoán hoàn tất!")
        
        # Best Prediction Display
        col_best = st.columns(1)[0]
        with col_best:
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 15px; border-radius: 10px; text-align: center;">
                <h3 style="color: white; margin: 0;">Kết quả Dự đoán</h3>
                <p style="color: #e0e0e0; margin: 5px 0;">Mô hình SVC</p>
                <p style="color: #FFD700; margin: 10px 0; font-size: 60px; font-weight: bold;">{predicted_class.upper()}</p>
            </div>
            """, unsafe_allow_html=True)

        # Class Info Display
        if predicted_class in CLASS_INFO:
            info = CLASS_INFO[predicted_class]
            st.markdown("### Thông tin Loài")
            st.markdown(f"""
            <div style="background-color: #1e1e2e; padding: 15px; border-radius: 10px; border: 1px solid #444;">
                <p style="margin: 5px 0;"><b>Tên khoa học:</b> {info['Scientific Name']}</p>
                <p style="margin: 5px 0;"><b>Kích thước:</b> {info['Size']}</p>
                <p style="margin: 5px 0;"><b>Môi trường sống:</b> {info['Habitat']}</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Stats
        st.subheader("Thống kê")
        col_stats1 = st.columns(1)[0]
        # with col_stats1:
        #     st.metric("Độ tin cậy", f"{confidence:.2f}%")
        with col_stats1:
            st.metric("Thời gian xử lý", f"{inference_time:.4f}s")
            
    elif uploaded_file is not None:
        st.error("Mô hình chưa được tải đúng cách. Vui lòng kiểm tra lại các file.")
    else:
        st.info("ℹ Tải ảnh lên để bắt đầu phân loại")
