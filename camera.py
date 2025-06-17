import cv2
import numpy as np
import streamlit as st
from PIL import Image
import time
from utils import detect_faces, predict_emotion_and_stress, get_stress_color

class CameraProcessor:
    def __init__(self, model):
        self.model = model
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        self.is_recording = False
        self.frame_count = 0
        self.last_prediction_time = 0
        self.prediction_interval = 1.0  # Prediksi setiap 1 detik
        
    def process_frame(self, frame):
        """
        Process single frame untuk deteksi dan prediksi
        """
        current_time = time.time()
        
        # Convert BGR to RGB for display
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Detect faces
        faces, gray = detect_faces(frame)
        
        results = []
        
        # Process each detected face
        for (x, y, w, h) in faces:
            # Draw rectangle around face
            cv2.rectangle(rgb_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # Extract face region
            face_roi = gray[y:y+h, x:x+w]
            
            # Predict emotion and stress level
            if current_time - self.last_prediction_time > self.prediction_interval:
                try:
                    prediction = predict_emotion_and_stress(self.model, face_roi)
                    results.append({
                        'bbox': (x, y, w, h),
                        'prediction': prediction
                    })
                    
                    # Add text overlay
                    emotion = prediction['emotion']
                    stress_label = prediction['stress_label']
                    confidence = prediction['emotion_confidence']
                    
                    # Get color based on stress level
                    stress_color = get_stress_color(prediction['stress_level'])
                    
                    # Convert hex color to RGB
                    stress_rgb = tuple(int(stress_color[i:i+2], 16) for i in (1, 3, 5))
                    
                    # Add text
                    text = f"{emotion} ({confidence:.2f})"
                    cv2.putText(rgb_frame, text, (x, y-10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, stress_rgb, 2)
                    cv2.putText(rgb_frame, stress_label, (x, y+h+20), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, stress_rgb, 2)
                    
                except Exception as e:
                    print(f"Prediction error: {e}")
        
        if current_time - self.last_prediction_time > self.prediction_interval:
            self.last_prediction_time = current_time
        
        return rgb_frame, results
    
    def process_uploaded_image(self, uploaded_file):
        """
        Process uploaded image file
        """
        try:
            # Read image
            image = Image.open(uploaded_file)
            
            # Convert to numpy array
            img_array = np.array(image)
            
            # Convert RGB to BGR for OpenCV
            if len(img_array.shape) == 3:
                img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            else:
                img_bgr = img_array
            
            # Process frame
            processed_frame, results = self.process_frame(img_bgr)
            
            return processed_frame, results
            
        except Exception as e:
            st.error(f"Error processing image: {e}")
            return None, []
    
    def start_webcam_stream(self):
        """
        Start webcam streaming dengan Streamlit
        """
        # Placeholder untuk webcam
        FRAME_WINDOW = st.image([])
        
        # Placeholder untuk hasil prediksi
        result_placeholder = st.empty()
        
        # Kontrol webcam
        col1, col2 = st.columns(2)
        with col1:
            start_button = st.button("Start Camera", key="start_cam")
        with col2:
            stop_button = st.button("Stop Camera", key="stop_cam")
        
        if start_button:
            self.is_recording = True
        if stop_button:
            self.is_recording = False
        
        if self.is_recording:
            # Buka webcam
            cap = cv2.VideoCapture(0)
            
            if not cap.isOpened():
                st.error("Cannot access webcam!")
                return
            
            # Loop untuk streaming
            while self.is_recording:
                ret, frame = cap.read()
                if not ret:
                    st.error("Failed to capture frame")
                    break
                
                # Process frame
                processed_frame, results = self.process_frame(frame)
                
                # Display frame
                FRAME_WINDOW.image(processed_frame, channels="RGB")
                
                # Display results
                if results:
                    self.display_results(results, result_placeholder)
                
                # Small delay to prevent overwhelming the browser
                time.sleep(0.1)
            
            cap.release()
            cv2.destroyAllWindows()
    
    def display_results(self, results, placeholder):
        """
        Display prediction results
        """
        if not results:
            return
        
        with placeholder.container():
            st.subheader("Hasil Deteksi:")
            
            for i, result in enumerate(results):
                prediction = result['prediction']
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write(f"**Wajah {i+1}:**")
                    st.write(f"Emosi: {prediction['emotion']}")
                    st.write(f"Confidence: {prediction['emotion_confidence']:.2f}")
                
                with col2:
                    stress_color = get_stress_color(prediction['stress_level'])
                    st.markdown(
                        f"<div style='padding: 10px; background-color: {stress_color}; "
                        f"border-radius: 5px; color: white; text-align: center;'>"
                        f"<strong>{prediction['stress_label']}</strong></div>",
                        unsafe_allow_html=True
                    )

def capture_photo_from_webcam():
    """
    Fungsi untuk mengambil foto dari webcam
    """
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        return None, "Cannot access webcam"
    
    ret, frame = cap.read()
    cap.release()
    
    if ret:
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return rgb_frame, None
    else:
        return None, "Failed to capture photo"

def save_captured_image(image, filename="captured_image.jpg"):
    """
    Simpan gambar yang di-capture
    """
    try:
        pil_image = Image.fromarray(image)
        pil_image.save(filename)
        return True, f"Image saved as {filename}"
    except Exception as e:
        return False, f"Error saving image: {e}"

class ImageProcessor:
    def __init__(self, model):
        self.model = model
        self.camera_processor = CameraProcessor(model)
    
    def process_single_image(self, image_source, source_type="upload"):
        """
        Process single image from various sources
        """
        if source_type == "upload":
            return self.camera_processor.process_uploaded_image(image_source)
        elif source_type == "webcam":
            return self.camera_processor.process_frame(image_source)
        else:
            raise ValueError("Invalid source type")
    
    def batch_process_images(self, image_list):
        """
        Process multiple images
        """
        results = []
        
        for i, image in enumerate(image_list):
            try:
                processed_frame, predictions = self.process_single_image(image)
                results.append({
                    'index': i,
                    'processed_frame': processed_frame,
                    'predictions': predictions
                })
            except Exception as e:
                results.append({
                    'index': i,
                    'error': str(e)
                })
        
        return results
    
    def get_face_statistics(self, results_list):
        """
        Analisis statistik dari hasil deteksi
        """
        if not results_list:
            return {}
        
        emotions = []
        stress_levels = []
        confidences = []
        
        for result in results_list:
            if 'predictions' in result and result['predictions']:
                for pred in result['predictions']:
                    prediction = pred['prediction']
                    emotions.append(prediction['emotion'])
                    stress_levels.append(prediction['stress_level'])
                    confidences.append(prediction['emotion_confidence'])
        
        if not emotions:
            return {}
        
        # Hitung statistik
        emotion_counts = {emotion: emotions.count(emotion) for emotion in set(emotions)}
        avg_stress = np.mean(stress_levels)
        avg_confidence = np.mean(confidences)
        
        return {
            'total_faces': len(emotions),
            'emotion_distribution': emotion_counts,
            'average_stress_level': avg_stress,
            'average_confidence': avg_confidence,
            'dominant_emotion': max(emotion_counts, key=emotion_counts.get)
        }

def create_demo_interface():
    """
    Membuat interface demo untuk testing
    """
    st.title("Stress Detection Demo")
    
    # Model loading (placeholder)
    if 'model' not in st.session_state:
        st.session_state.model = None
    
    if st.session_state.model is None:
        st.warning("Model not loaded. Please load a trained model first.")
        return
    
    # Tab selection
    tab1, tab2, tab3 = st.tabs(["Upload Image", "Webcam", "Batch Process"])
    
    with tab1:
        st.header("Upload Image")
        uploaded_file = st.file_uploader(
            "Choose an image...", 
            type=['jpg', 'jpeg', 'png']
        )
        
        if uploaded_file is not None:
            processor = ImageProcessor(st.session