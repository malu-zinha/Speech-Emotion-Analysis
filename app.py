import streamlit as st
import numpy as np
import librosa
import joblib
import tensorflow as tf
import os
import soundfile as sf

# --------------------------------- PARTE 1: EXTRAIR FEATURES --------------------------------- #

# Carregar o modelo e o scaler
MODEL_PATH = "models/audio_emotion_model.keras"  # Example
SCALER_PATH = "models/scaler.pkl"                # Example

model = tf.keras.models.load_model(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# Lista de emoções
EMOTIONS = ["angry", "calm", "disgust", "fear",
            "happy", "neutral", "sad", "surprise"]


# Função para extrair features
def extract_features(audio_path):
    data, sr = librosa.load(audio_path, sr=16000, mono=True)
    features = []

    # Zero Crossing Rate
    # Extract the zcr here
    # features.extend(zcr)

    # Chroma STFT
    # Extract the chroma stft here
    # features.extend(chroma)

    # MFCCs
    # Extract the mfccs here
    # features.extend(mfccs)

    # RMS
    # Extract the rms here
    # features.extend(rms)

    # Mel Spectrogram
    # Extract the mel here
    # features.extend(mel)

    # Garantir que tenha exatamente 162 features (ou truncar/zerar)
    target_length = 162
    if len(features) < target_length:
        features.extend([0] * (target_length - len(features)))
    elif len(features) > target_length:
        features = features[:target_length]

    return np.array(features).reshape(1, -1)


# --------------------------------- PARTE 2: STREAMLIT --------------------------------- #

# Configuração do app Streamlit (Título e descrição)
# Code here

# Upload de arquivo de áudio (wav, mp3, ogg)
uploaded_file = st.file_uploader(
    "Escolha um arquivo de áudio...", type=["wav", "mp3", "ogg"])

if uploaded_file is not None:
    # Salvar temporariamente o áudio
    # Code here

    # Reproduzir o áudio enviado
    # Code here

    # Extrair features
    # Code here

    # Normalizar os dados com o scaler treinado
    # Code here

    # Ajustar formato para o modelo
    # Code here

    # Fazer a predição
    # Code here

    # Exibir o resultado
    # Code here

    # Exibir probabilidades (gráfico de barras)
    # Code here

    # Remover o arquivo temporário
    # Code here
