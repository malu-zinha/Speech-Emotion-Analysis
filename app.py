import streamlit as st
import numpy as np
import librosa
import joblib
import tensorflow as tf
import os
import soundfile as sf

# Carregar o modelo e o scaler
MODEL_PATH = "models/audio_emotion_model.keras"
SCALER_PATH = "models/scaler.pkl"

model = tf.keras.models.load_model(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# Lista de emoções
EMOTIONS = ["angry", "calm", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

# Função para extrair features
def extract_features(audio_path):
    data, sr = librosa.load(audio_path, sr=16000, mono=True)
    features = []

    # Zero Crossing Rate
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=data).T, axis=0)
    features.extend(zcr)

    # Chroma STFT
    chroma = np.mean(librosa.feature.chroma_stft(y=data, sr=sr).T, axis=0)
    features.extend(chroma)

    # MFCCs 
    mfccs = np.mean(librosa.feature.mfcc(y=data, sr=sr, n_mfcc=13).T, axis=0)
    features.extend(mfccs)

    # RMS
    rms = np.mean(librosa.feature.rms(y=data).T, axis=0)
    features.extend(rms)

    # Mel Spectrogram 
    mel = np.mean(librosa.feature.melspectrogram(y=data, sr=sr, n_mels=128).T, axis=0)
    features.extend(mel)

    # Garantir que tenha exatamente 162 features (ou truncar/zerar)
    target_length = 162
    if len(features) < target_length:
        features.extend([0] * (target_length - len(features))) 
    elif len(features) > target_length:
        features = features[:target_length]  

    return np.array(features).reshape(1, -1)

# Configuração do app
st.title("Detector de Emoções em Áudio 🎵")
st.write("Envie um arquivo de áudio para análise!")

# Upload de arquivo de áudio
uploaded_file = st.file_uploader("Escolha um arquivo de áudio...", type=["wav", "mp3", "ogg"])

if uploaded_file is not None:
    # Salvar temporariamente o áudio
    temp_audio_path = "temp_audio.wav"
    with open(temp_audio_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Reproduzir o áudio enviado
    st.audio(temp_audio_path, format="audio/wav")

    # Extrair features
    features = extract_features(temp_audio_path)

    # Normalizar os dados com o scaler treinado
    features_scaled = scaler.transform(features)

    # Ajustar formato para o modelo
    features_scaled = np.expand_dims(features_scaled, axis=2)

    # Fazer a previsão
    prediction = model.predict(features_scaled)
    predicted_emotion = EMOTIONS[np.argmax(prediction)]

    # Exibir o resultado
    st.subheader("🎭 Emoção Detectada:")
    st.write(f"**{predicted_emotion.upper()}**")

    # Exibir probabilidades
    st.bar_chart(prediction[0])

    # Remover o arquivo temporário
    os.remove(temp_audio_path)
