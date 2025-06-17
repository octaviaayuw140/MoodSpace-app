import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tempfile
import os
from utils import (
    detect_faces, predict_emotion_and_stress, load_and_validate_model,
    get_stress_color, get_recommendations, enhance_image_quality,EMOTION_LABELS, STRESS_LEVELS
)

# Page configuration
st.set_page_config(
    page_title="Stress Detection App",
    page_icon="😌",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .stress-card {
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        margin: 1rem 0;
    }
    .emotion-bar {
        margin: 0.5rem 0;
        padding: 0.5rem;
        border-radius: 5px;
        background-color: #f0f2f6;
    }
    .recommendation-box {
        background-color: #e8f4f8;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #00bcd4;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🧠 AI Stress Detection System</h1>
        <p>Deteksi tingkat stress real-time menggunakan analisis ekspresi wajah</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Pengaturan")
        
        # Model selection
        model_path = st.text_input(
            "Path Model ().keras)",
            value="emotion_model.keras",
            help="Masukkan path ke file model yang sudah dilatih"
        )
        
        # Detection settings
        st.subheader("Pengaturan Deteksi")
        confidence_threshold = st.slider(
            "Confidence Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.1
        )
        
        # Info section
        st.subheader("ℹ️ Informasi")
        st.info("""
        **Cara Penggunaan:**
        1. Upload foto atau gunakan webcam
        2. Sistem akan mendeteksi wajah
        3. Analisis emosi dan stress level
        4. Dapatkan rekomendasi
        """)
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("📸 Input Gambar")
        
        # Input methods
        input_method = st.radio(
            "Pilih metode input:",
            ["Upload Gambar", "Webcam", "Kamera Real-time"]
        )
        
        if input_method == "Upload Gambar":
            uploaded_file = st.file_uploader(
                "Upload gambar",
                type=['jpg', 'jpeg', 'png', 'bmp'],
                help="Upload gambar dengan wajah yang jelas"
            )
            
            if uploaded_file is not None:
                image = Image.open(uploaded_file)
                image_array = np.array(image)
                
                # Convert RGB to BGR for OpenCV
                if len(image_array.shape) == 3:
                    image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
                
                process_image(image_array, model_path, confidence_threshold)
        
        elif input_method == "Webcam":
            camera_input = st.camera_input("Ambil foto dengan webcam")
            
            if camera_input is not None:
                image = Image.open(camera_input)
                image_array = np.array(image)
                
                # Convert RGB to BGR for OpenCV
                image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
                
                process_image(image_array, model_path, confidence_threshold)
        
        elif input_method == "Kamera Real-time":
            st.info("🚧 Fitur real-time camera sedang dalam pengembangan")
            st.markdown("""
            Untuk menggunakan real-time detection, jalankan:
            ```bash
            python camera.py
            ```
            """)
    
    with col2:
        # Emotion labels info
        st.subheader("🎭 Label Emosi")
        for idx, emotion in EMOTION_LABELS.items():
            st.write(f"**{idx}:** {emotion.title()}")
        
        # Stress levels info
        st.subheader("📈 Level Stress")
        for idx, level in STRESS_LEVELS.items():
            color = get_stress_color(idx)
            st.markdown(f"""
            <div style="background-color: {color}; padding: 0.2rem 0.5rem; 
                        border-radius: 5px; margin: 0.2rem 0; color: white;">
                <strong>{idx}:</strong> {level}
            </div>
            """, unsafe_allow_html=True)

def process_image(image, model_path, confidence_threshold):
    """Process uploaded image for emotion and stress detection"""
    
    # Load model
    if not os.path.exists(model_path):
        st.error("Model file tidak ditemukan!")
        return
    
    model, status = load_and_validate_model(model_path)
    if model is None:
        st.error(f"Error loading model: {status}")
        return
    
    # Display original image
    st.subheader("📷 Gambar Asli")
    display_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    st.image(display_image, use_column_width=True)
    
    # Detect faces
    with st.spinner("Mendeteksi wajah..."):
        enchanced_image = enhance_image_quality(image)
        faces, gray = detect_faces(image)
    
    if len(faces) == 0:
        st.warning("⚠️ Tidak ada wajah yang terdeteksi")
        st.info("Tips: Pastikan wajah terlihat jelas dan pencahayaan cukup")
        return
    
    st.success(f"✅ Terdeteksi {len(faces)} wajah")
    
    # Process each face
    for i, (x, y, w, h) in enumerate(faces):
        st.subheader(f"👤 Analisis Wajah {i+1}")
        
        # Extract face
        face_image = gray[y:y+h, x:x+w]
        
        # Show detected face
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Wajah Terdeteksi:**")
            st.image(face_image, width=200, channels="GRAY")
        
        with col2:
            # Predict emotion and stress
            with st.spinner("Menganalisis emosi dan stress..."):
                result = predict_emotion_and_stress(model, face_image)
            
            if result is None:
                st.error("❌ Gagal menganalisis wajah")
                continue
            
            # Display results
            display_results(result, confidence_threshold)
        
        # Draw rectangle on original image
        cv2.rectangle(display_image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # Add emotion label
        emotion_text = f"{result['emotion']} ({result['emotion_confidence']:.2f})"
        cv2.putText(display_image, emotion_text, (x, y-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Show annotated image
    st.subheader("🎯 Hasil Deteksi")
    st.image(display_image, use_column_width=True)

def display_results(result, confidence_threshold):
    """Display emotion and stress detection results"""
    
    # Main emotion result
    emotion = result['emotion']
    confidence = result['emotion_confidence']
    
    if confidence >= confidence_threshold:
        st.success(f"**Emosi Dominan:** {emotion.title()}")
        st.write(f"**Confidence:** {confidence:.2%}")
    else:
        st.warning(f"**Emosi:** {emotion.title()} (Confidence rendah: {confidence:.2%})")
    
    # Stress level
    stress_level = result['stress_level']
    stress_label = result['stress_label']
    stress_color = get_stress_color(stress_level)
    
    st.markdown(f"""
    <div class="stress-card" style="background-color: {stress_color}; color: white;">
        <h3>Tingkat Stress: {stress_label}</h3>
        <p>Level: {stress_level}/4</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Emotion probabilities
    st.subheader("📊 Distribusi Emosi")
    all_predictions = result['all_predictions']
    
    for emotion_name, probability in all_predictions.items():
        st.markdown(f"""
        <div class="emotion-bar">
            <strong>{emotion_name.title()}:</strong> {probability:.2%}
            <div style="background-color: #ddd; border-radius: 5px; margin-top: 5px;">
                <div style="background-color: #4CAF50; width: {probability*100}%; 
                           height: 20px; border-radius: 5px;"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Recommendations
    st.subheader("💡 Rekomendasi")
    recommendations = get_recommendations(stress_level)
    
    for i, rec in enumerate(recommendations, 1):
        st.markdown(f"""
        <div class="recommendation-box">
            <strong>{i}.</strong> {rec}
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()