import logging
import queue
import os
from datetime import datetime
from typing import List, NamedTuple

import av
import cv2
import numpy as np
import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode
import tensorflow as tf

# --- Logging ---
logger = logging.getLogger(__name__)

# --- Emotion Labels & Emoji ---
EMOTION_LABELS = ['happy', 'sad', 'angry', 'fear', 'surprise', 'disgust', 'neutral']
EMOJI_MAP = {
    'happy': '😊',
    'sad': '😢',
    'angry': '😠',
    'fear': '😨',
    'surprise': '😲',
    'disgust': '🤢',
    'neutral': '😐'
}

# --- Streamlit UI ---
st.set_page_config(page_title="Stress Detection", layout="centered")
st.markdown("""
<div style="text-align: center;">
    <h1>🧠 AI Stress Detection System</h1>
    <h3>Deteksi Tingkat Stress Melalui Ekspresi Wajah</h3>
    <p>Menggunakan Deep Learning & Computer Vision untuk analisis emosi real-time</p>
</div>
""", unsafe_allow_html=True)

# --- Load Model ---
@st.cache_resource
def load_stress_model():
    model_path = "test-model.h5"
    if not os.path.exists(model_path):
        st.error(f"Model tidak ditemukan di: {model_path}")
        return None
    try:
        return tf.keras.models.load_model(model_path, compile=False)
    except Exception as e:
        st.error(f"Gagal memuat model: {str(e)}")
        return None

model = load_stress_model()
result_queue: "queue.Queue[List[str]]" = queue.Queue()

class Detection(NamedTuple):
    emotion: str
    score: float

# --- Webcam Frame Callback ---
def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
    image = frame.to_ndarray(format="bgr24")
    frame = cv2.flip(image, 1)

    h, w, _ = frame.shape
    box_size = 250
    x1, y1 = 20, 60
    x2, y2 = x1 + box_size, y1 + box_size

    roi = frame[y1:y2, x1:x2]
    predicted_emotion = "unknown"
    score = 0.0

    if model is not None:
        try:
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            resized = cv2.resize(gray, (48, 48))
            normalized = resized / 255.0
            reshaped = normalized.reshape(1, 48, 48, 1)
            preds = model.predict(reshaped, verbose=0)
            score = float(np.max(preds))
            predicted_emotion = EMOTION_LABELS[np.argmax(preds)]
        except Exception as e:
            logger.warning(f"Prediction error: {e}")
    else:
        logger.warning("Model not loaded.")

    emoji = EMOJI_MAP.get(predicted_emotion, '🤔')
    text = f'{predicted_emotion.upper()} {emoji} ({score:.2f})'

    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(frame, text, (x1, y2 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    cv2.putText(frame, 'Arahkan wajah ke dalam kotak', (x1, y2 + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Clear previous result
    while not result_queue.empty():
        result_queue.get()
    result_queue.put([Detection(emotion=predicted_emotion, score=score)])

    return av.VideoFrame.from_ndarray(frame, format="bgr24")

# --- WebRTC Stream ---
webrtc_ctx = webrtc_streamer(
    key="stress-detection",
    mode=WebRtcMode.SENDRECV,
    video_frame_callback=video_frame_callback,
    media_stream_constraints={
        "video": {
            "width": {"ideal": 640},
            "height": {"ideal": 480},
            "frameRate": {"ideal": 15}
        },
        "audio": False
    },
    async_processing=True,
)

# --- Tampilkan Hasil Deteksi ---
if st.checkbox("Tampilkan hasil deteksi", value=True):
    if webrtc_ctx.state.playing and not result_queue.empty():
        result = result_queue.get()
        st.table([r._asdict() for r in result])
    elif model is None:
        st.warning("Model belum berhasil dimuat.")
