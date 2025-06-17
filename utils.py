import cv2
import numpy as np
import tensorflow as tf
from keras.models import load_model
import os

# Emotion labels (sesuai dengan FER2013 dataset)
EMOTION_LABELS = {
    0: 'angry',
    1: 'disgust', 
    2: 'fear',
    3: 'happy',
    4: 'sad',
    5: 'surprise',
    6: 'neutral'
}

# Stress level mapping
STRESS_LEVELS = {
    0: 'Very Low',
    1: 'Low', 
    2: 'Moderate',
    3: 'High',
    4: 'Very High'
}

def detect_faces(image):
    """
    Detect faces in an image using OpenCV Haar Cascade
    
    Args:
        image: Input image (BGR format)
    
    Returns:
        faces: List of face coordinates [(x, y, w, h), ...]
        gray: Grayscale version of input image
    """
    try:
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Load face cascade
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Detect faces with optimized parameters
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        return faces, gray
        
    except Exception as e:
        print(f"Error in face detection: {e}")
        return [], image

def preprocess_face(face_image, target_size=(48, 48)):
    """
    Preprocess face image for emotion prediction
    
    Args:
        face_image: Input face image (grayscale)
        target_size: Target size for model input
    
    Returns:
        processed_face: Preprocessed face array ready for prediction
    """
    try:
        # Resize to target size
        face_resized = cv2.resize(face_image, target_size)
        
        # Normalize pixel values to [0, 1]
        face_normalized = face_resized.astype('float32') / 255.0
        
        # Add batch dimension and channel dimension
        face_processed = np.expand_dims(face_normalized, axis=0)  # Batch dimension
        face_processed = np.expand_dims(face_processed, axis=-1)  # Channel dimension
        
        return face_processed
        
    except Exception as e:
        print(f"Error in face preprocessing: {e}")
        return None

def predict_emotion_and_stress(model, face_image):
    """
    Predict emotion and calculate stress level from face image
    
    Args:
        model: Trained emotion recognition model
        face_image: Input face image (grayscale)
    
    Returns:
        prediction_dict: Dictionary containing emotion, stress level, and confidence scores
    """
    try:
        # Preprocess face
        processed_face = preprocess_face(face_image)
        
        if processed_face is None:
            return None
        
        # Make prediction
        predictions = model.predict(processed_face, verbose=0)
        emotion_probabilities = predictions[0]
        
        # Get predicted emotion
        predicted_emotion_idx = np.argmax(emotion_probabilities)
        predicted_emotion = EMOTION_LABELS[predicted_emotion_idx]
        emotion_confidence = float(emotion_probabilities[predicted_emotion_idx])
        
        # Calculate stress level based on emotion probabilities
        stress_level = calculate_stress_level(emotion_probabilities)
        stress_label = STRESS_LEVELS[stress_level]
        
        # Create all predictions dictionary
        all_predictions = {}
        for idx, prob in enumerate(emotion_probabilities):
            emotion_name = EMOTION_LABELS[idx]
            all_predictions[emotion_name] = float(prob)
        
        return {
            'emotion': predicted_emotion,
            'emotion_confidence': emotion_confidence,
            'stress_level': stress_level,
            'stress_label': stress_label,
            'all_predictions': all_predictions,
            'raw_predictions': emotion_probabilities.tolist()
        }
        
    except Exception as e:
        print(f"Error in emotion prediction: {e}")
        return None

def calculate_stress_level(emotion_probabilities):
    """
    Calculate stress level based on emotion probabilities
    
    Args:
        emotion_probabilities: Array of emotion probabilities
    
    Returns:
        stress_level: Integer stress level (0-4)
    """
    try:
        # Emotion to stress mapping
        # Higher values indicate more stress-inducing emotions
        stress_weights = {
            0: 0.9,   # angry - high stress
            1: 0.7,   # disgust - medium-high stress
            2: 0.8,   # fear - high stress
            3: 0.1,   # happy - low stress
            4: 0.6,   # sad - medium stress
            5: 0.4,   # surprise - medium-low stress
            6: 0.2    # neutral - low stress
        }
        
        # Calculate weighted stress score
        weighted_stress = 0.0
        for emotion_idx, probability in enumerate(emotion_probabilities):
            weighted_stress += probability * stress_weights[emotion_idx]
        
        # Convert to stress level (0-4)
        if weighted_stress <= 0.2:
            return 0  # Very Low
        elif weighted_stress <= 0.4:
            return 1  # Low
        elif weighted_stress <= 0.6:
            return 2  # Moderate
        elif weighted_stress <= 0.8:
            return 3  # High
        else:
            return 4  # Very High
            
    except Exception as e:
        print(f"Error calculating stress level: {e}")
        return 2  # Default to moderate

def get_stress_color(stress_level):
    """
    Get color code for stress level visualization
    
    Args:
        stress_level: Integer stress level (0-4)
    
    Returns:
        color_hex: Hex color code
    """
    colors = {
        0: '#4CAF50',  # Green - Very Low
        1: '#8BC34A',  # Light Green - Low
        2: '#FFC107',  # Yellow - Moderate
        3: '#FF9800',  # Orange - High
        4: '#F44336'   # Red - Very High
    }
    
    return colors.get(stress_level, '#FFC107')

def get_recommendations(stress_level):
    """
    Get stress management recommendations based on stress level
    
    Args:
        stress_level: Integer stress level (0-4)
    
    Returns:
        recommendations: List of recommendation strings
    """
    recommendation_map = {
        0: [  # Very Low Stress
            "Maintain your current positive state with regular self-care activities.",
            "Continue with your healthy habits and routines.",
            "Consider helping others or engaging in creative activities.",
            "Take time to appreciate and celebrate your well-being."
        ],
        1: [  # Low Stress
            "Keep up your good stress management practices.",
            "Engage in light physical activities like walking or stretching.",
            "Practice gratitude and mindfulness for a few minutes daily.",
            "Maintain good sleep hygiene and regular meal times."
        ],
        2: [  # Moderate Stress
            "Take short breaks throughout the day to relax and recharge.",
            "Practice deep breathing exercises for 5-10 minutes.",
            "Consider light exercise like yoga or a short walk.",
            "Try progressive muscle relaxation techniques.",
            "Limit caffeine intake and ensure adequate hydration."
        ],
        3: [  # High Stress
            "Take immediate steps to reduce your stress levels.",
            "Practice deep breathing: inhale for 4 counts, hold for 4, exhale for 4.",
            "Try the 5-4-3-2-1 grounding technique: identify 5 things you see, 4 you hear, 3 you touch, 2 you smell, 1 you taste.",
            "Take a 10-15 minute break from your current activity.",
            "Consider talking to someone you trust about what's causing stress.",
            "Engage in physical activity to release tension."
        ],
        4: [  # Very High Stress
            "IMMEDIATE ACTION NEEDED - Take steps to address your stress now.",
            "Stop what you're doing and focus on slow, deep breathing for 5 minutes.",
            "Remove yourself from the stressful situation if possible.",
            "Contact a friend, family member, or mental health professional for support.",
            "Practice emergency stress relief: splash cold water on your face, listen to calming music.",
            "Consider seeking professional help if this level of stress persists.",
            "Avoid making important decisions while in this state.",
            "Focus on basic self-care: eat something nutritious, drink water, rest."
        ]
    }
    
    return recommendation_map.get(stress_level, recommendation_map[2])

def save_detection_result(result, save_dir="detection_logs"):
    """
    Save detection results to file for later analysis
    
    Args:
        result: Detection result dictionary
        save_dir: Directory to save results
    """
    try:
        # Create directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)
        
        # Generate filename with timestamp
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"detection_{timestamp}.json"
        filepath = os.path.join(save_dir, filename)
        
        # Save to JSON file
        import json
        with open(filepath, 'w') as f:
            json.dump(result, f, indent=2, default=str)
        
        print(f"Detection result saved to {filepath}")
        
    except Exception as e:
        print(f"Error saving detection result: {e}")

def load_and_validate_model(model_path):
    """
    Load and validate emotion recognition model
    
    Args:
        model_path: Path to the model file
    
    Returns:
        model: Loaded model or None if failed
        status: Status message
    """
    try:
        if not os.path.exists(model_path):
            return None, f"File tidak ditemukan: {model_path}"
        
        # Coba load dengan format Keras
        try:
            model = load_model(model_path, compile=False)
        except:
            # Coba load dengan format H5 lama
            model = load_model(model_path, compile=False)
        
        # Validate model input shape
        expected_shape = (None, 48, 48, 1)
        actual_shape = model.input_shape
        
        if actual_shape != expected_shape:
            return None, f"Model input shape mismatch. Expected {expected_shape}, got {actual_shape}"
        
        # Test prediction with dummy data
        dummy_input = np.random.random((1, 48, 48, 1))
        test_prediction = model.predict(dummy_input, verbose=0)
        
        if test_prediction.shape[1] != 7:
            return None, f"Model output shape mismatch. Expected 7 emotions, got {test_prediction.shape[1]}"
        
        return model, "Model loaded and validated successfully"
        
    except Exception as e:
        import traceback
        return None, f"Error loading model: {str(e)}\n{traceback.format_exc()}"

def enhance_image_quality(image):
    """
    Enhance image quality for better face detection
    
    Args:
        image: Input image
    
    Returns:
        enhanced_image: Enhanced image
    """
    try:
        # Convert to grayscale if colored
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Apply histogram equalization
        enhanced = cv2.equalizeHist(gray)
        
        # Apply Gaussian blur to reduce noise
        enhanced = cv2.GaussianBlur(enhanced, (3, 3), 0)
        
        return enhanced
        
    except Exception as e:
        print(f"Error in image enhancement: {e}")
        return image

def validate_face_quality(face_image, min_size=(30, 30)):
    """
    Validate if the detected face is suitable for emotion recognition
    
    Args:
        face_image: Detected face image
        min_size: Minimum acceptable face size
    
    Returns:
        is_valid: Boolean indicating if face is valid
        quality_score: Quality score (0-1)
    """
    try:
        # Check minimum size
        height, width = face_image.shape[:2]
        if height < min_size[0] or width < min_size[1]:
            return False, 0.0
        
        # Calculate quality metrics
        # 1. Check brightness
        brightness = np.mean(face_image)
        brightness_score = 1.0 if 50 <= brightness <= 200 else max(0, 1 - abs(brightness - 125) / 125)
        
        # 2. Check contrast
        contrast = np.std(face_image)
        contrast_score = min(1.0, contrast / 50)
        
        # 3. Check for blur (using Laplacian variance)
        blur_score = cv2.Laplacian(face_image, cv2.CV_64F).var()
        blur_score = min(1.0, blur_score / 100)
        
        # Calculate overall quality score
        quality_score = (brightness_score + contrast_score + blur_score) / 3
        
        # Face is valid if quality score > 0.3
        is_valid = quality_score > 0.3
        
        return is_valid, quality_score
        
    except Exception as e:
        print(f"Error in face quality validation: {e}")
        return True, 0.5  # Default to valid with medium quality