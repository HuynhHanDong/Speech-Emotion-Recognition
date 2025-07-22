import streamlit as st
import numpy as np
import tempfile
import os
import librosa
import tensorflow as tf

SAMPLE_RATE = 22050
DURATION = 3
TIME_STEPS = 3
FRAME_PER_STEP = 60
N_MELS = 128
MODEL = "MardeusNet.keras"  # Change model path if needed
class_names = ['angry', 'disgust', 'fearful', 'happy', 'neutral', 'sad', 'surprised']

class FocalLoss(tf.keras.losses.Loss):
    def __init__(self, gamma=2.0, alpha=None, reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE, name='focal_loss'):
        super(FocalLoss, self).__init__(reduction=reduction, name=name)
        self.gamma = gamma
        self.alpha = alpha  # Now can be list or tensor

    def call(self, y_true, y_pred):
        epsilon = 1e-7
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)

        cross_entropy = -y_true * tf.math.log(y_pred)
        weight = tf.pow(1 - y_pred, self.gamma)

        if self.alpha is not None:
            # assume alpha is a list/array of class weights
            alpha = tf.constant(self.alpha, dtype=tf.float32)
            alpha = tf.reshape(alpha, (1, -1))  # shape (1, num_classes)
            alpha_factor = y_true * alpha  # broadcast across batch
            focal_loss = alpha_factor * weight * cross_entropy
        else:
            focal_loss = weight * cross_entropy

        return tf.reduce_mean(tf.reduce_sum(focal_loss, axis=-1))

    def get_config(self):
        config = super().get_config()
        config.update({
            "gamma": self.gamma,
            "alpha": self.alpha,
        })
        return config

def audio_to_melspectrogram(audio, sr=SAMPLE_RATE):
    spectrogram = librosa.feature.melspectrogram(
        y=audio, 
        sr=sr,
        n_mels=N_MELS,
        n_fft=2048,
        hop_length=512
    )
    spectrogram = librosa.power_to_db(spectrogram, ref=np.max)
    spectrogram = (spectrogram - np.min(spectrogram)) / (np.max(spectrogram) - np.min(spectrogram) + 1e-8)
    return spectrogram

def load_and_process_audio(file_path):
    audio, sr = librosa.load(file_path, sr=SAMPLE_RATE, duration=DURATION)

    if len(audio) < SAMPLE_RATE * DURATION:
        audio = np.pad(audio, (0, SAMPLE_RATE * DURATION - len(audio)))

    chunk_len = len(audio) // TIME_STEPS
    segments = []

    for i in range(TIME_STEPS):
        chunk = audio[i * chunk_len : (i + 1) * chunk_len]

        spectrogram = audio_to_melspectrogram(chunk, sr)

        if spectrogram.shape[1] < FRAME_PER_STEP:
            spectrogram = np.pad(spectrogram, ((0, 0), (0, FRAME_PER_STEP - spectrogram.shape[1])))
        else:
            spectrogram = spectrogram[:, :FRAME_PER_STEP]

        spectrogram = spectrogram[..., np.newaxis]  # Add channel
        segments.append(spectrogram)

    return np.stack(segments, axis=0)  # (time_steps, n_mels, frames, 1)

@st.cache_resource
def load_my_model():
    return tf.keras.models.load_model(MODEL, custom_objects={"FocalLoss": FocalLoss})

model = load_my_model()

# ====== Streamlit UI ======
st.title("🎵 Audio Classification App")
st.write("Upload an audio file to classify it using the trained model.")

st.title("🎙️ Audio Classifier")
input_choice = st.radio("Choose input method:", ["Upload Audio", "Record Audio"])
audio_bytes = None

if input_choice == "Upload Audio":
    uploaded_file = st.file_uploader("Upload an audio file", type=["wav", "mp3", "ogg"])
    if uploaded_file:
        st.audio(uploaded_file)
        audio_bytes = uploaded_file.read()

elif input_choice == "Record Audio":
    recorded_audio = st.audio_input("Record your voice")
    if recorded_audio:
        st.audio(recorded_audio)
        audio_bytes = recorded_audio.read()

if audio_bytes:  
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
        tmp_file.write(audio_bytes)
        tmp_file_path = tmp_file.name

    st.write("🔄 Processing audio...")
    processed_audio = load_and_process_audio(tmp_file_path)  # shape: (TIME_STEPS, n_mels, frames, 1)

    # Expand dims if model expects batch
    processed_audio = np.expand_dims(processed_audio, axis=0)  # shape: (1, TIME_STEPS, n_mels, frames, 1)

    # Predict
    st.write("🔍 Making prediction...")
    prediction = model.predict(processed_audio)

    # Output result
    predicted_index = np.argmax(prediction)
    predicted_class = class_names[predicted_index]
    confidence = prediction[0][predicted_index]
    st.success(f"✅ Predicted class: {predicted_class}")
    st.write("Confidence scores:", confidence)