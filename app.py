import streamlit as st
import numpy as np
import librosa
import joblib
import tensorflow as tf
import os
import tempfile
import pandas as pd
import plotly.express as px
import tempfile

# --------------------------------- PARTE 1: EXTRAIR FEATURES --------------------------------- #

# Carregando o modelo e o scaler:
MODEL_PATH = "notebooks/models/fashion_model.keras"
SCALER_PATH = "notebooks/models/scaler.pkl"

model = tf.keras.models.load_model(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# Lista de emoções:
EMOTIONS = ["angry", "calm", "disgust", "fear",
            "happy", "neutral", "sad", "surprise"]

# Traduzindo a lista de emoções para português:
EMOTIONS_PT = {
    "angry": "Raiva",
    "calm": "Calma", 
    "disgust": "Nojo",
    "fear": "Medo",
    "happy": "Felicidade",
    "neutral": "Neutra",
    "sad": "Tristeza",
    "surprise": "Surpresa"
}

EMOTIONS_PT_EMOJI = {
    "angry": "Raiva 😡",
    "calm": "Calma 😌", 
    "disgust": "Nojo 🤢",
    "fear": "Medo 🥶",
    "happy": "Felicidade 😄",
    "neutral": "Neutra 😐",
    "sad": "Tristeza 😭",
    "surprise": "Surpresa 😮"
}   

def extract_features(audio_path): # Função para extrair features:

    data, sample_rate = librosa.load(audio_path, duration=2.5, offset=0.6)
    result = np.array([])

    # Zero Crossing Rate:
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=data).T, axis=0)
    result = np.hstack((result, zcr))

    # Chroma_stft:
    chroma_stft = np.mean(librosa.feature.chroma_stft(y=data, sr=sample_rate).T, axis=0)
    result = np.hstack((result, chroma_stft))

    # MFCC:
    mfcc = np.mean(librosa.feature.mfcc(y=data, sr=sample_rate, n_mfcc=40).T, axis=0)
    result = np.hstack((result, mfcc))

    # Root Mean Square Value:
    rms = np.mean(librosa.feature.rms(y=data).T, axis=0)
    result = np.hstack((result, rms))

    # MelSpectrogram:
    mel = np.mean(librosa.feature.melspectrogram(y=data, sr=sample_rate).T, axis=0)
    result = np.hstack((result, mel))
    
    # Garantindo que tenha exatamente 182 features (ou truncar/zerar):
    target_length = 182
    if len(result) < target_length:
        result = np.pad(result, (0, target_length - len(result)), 'constant')
    elif len(result) > target_length:
        result = result[:target_length]
    
    return result.reshape(1, -1)


# --------------------------------- PARTE 2: STREAMLIT --------------------------------- #

# Configuração do app Streamlit (Título e descrição):
st.title("Speech-Emotion-Analysis:")
st.write("Este aplicativo reconhece emoções em arquivos de áudio.")

# Upload de arquivo de áudio (wav, mp3, ogg):
uploaded_file = st.file_uploader("Escolha um arquivo de áudio...", type=["wav", "mp3", "ogg"])

if uploaded_file is not None:

    # Salvando temporariamente o áudio:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
        tmp_file.write(uploaded_file.read())
        temp_audio_path = tmp_file.name

    # Reproduzindo o áudio enviado:
    st.audio(uploaded_file, format='audio/wav')

    # Extraindo features:
    features = extract_features(temp_audio_path)

    # Normalizando os dados com o scaler treinado:
    features = scaler.transform(features)

    # Ajustando formato para o modelo:
    features = features.reshape(1, -1)

    # Fazendo a predição de emoções:
    prediction = model.predict(features)
    predicted_class = np.argmax(prediction, axis=1)[0]
    predicted_emotion = EMOTIONS[predicted_class]

    # Exibindo o resultado:
    predicted_emotion_pt = EMOTIONS_PT_EMOJI[predicted_emotion]
    st.success(f"🎭 Emoção reconhecida: **{predicted_emotion_pt.upper()}**")
    
    # Exibindo as probabilidades:
    probabilities = prediction[0] 
    probabilities_norm = probabilities / np.sum(probabilities) * 100
    
    # Criando DataFrame com emoções:
    emotions_pt = [EMOTIONS_PT[emotion] for emotion in EMOTIONS]
    emotion_data = pd.DataFrame({
        'Emoção': emotions_pt,
        'Probabilidade (%)': np.round(probabilities_norm, 1)
    })

    emotion_colors = ['#fe88b1', '#66c5cc', '#8be0a4', '#b097e7', '#f6cf71', '#cccccc', '#9eb9f3', '#f89c74']

    st.subheader("📊 Probabilidades de cada emoção:")
    fig = px.bar(
        emotion_data, 
        x='Emoção', 
        y='Probabilidade (%)',
        color='Emoção',
        color_discrete_sequence=emotion_colors,
        title="Análise de Emoções"
    )
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig)

    # Removendo o arquivo temporário:
    os.remove(temp_audio_path)
