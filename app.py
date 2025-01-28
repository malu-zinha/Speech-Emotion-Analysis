import streamlit as st
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import pickle

# Carregar o modelo treinado
MODEL_PATH = "models/audio_emotion_model.pkl"
with open(MODEL_PATH, "rb") as file:
    model = pickle.load(file)

# Função para extrair features do áudio
def extract_features(audio_file):
    audio, sample_rate = librosa.load(audio_file, res_type='kaiser_fast')
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    mfccs_scaled = np.mean(mfccs.T, axis=0)
    return mfccs_scaled.reshape(1, -1)

# Configuração da página
st.title("Reconhecimento de Emoções em Áudio")
st.write("Envie um arquivo de áudio para identificar a emoção presente nele.")

# Upload do arquivo de áudio
uploaded_file = st.file_uploader("Envie seu arquivo de áudio (.wav, .mp3)", type=["wav", "mp3"])

if uploaded_file is not None:
    # Exibir informações do arquivo enviado
    st.audio(uploaded_file, format='audio/wav')
    
    # Extração de features
    try:
        features = extract_features(uploaded_file)
        
        # Predição da emoção
        prediction = model.predict(features)
        probabilities = model.predict_proba(features)
        
        # Exibir resultado
        st.write("### Emoção Detectada:", prediction[0].capitalize())
        
        # Exibir probabilidades
        st.write("### Probabilidades de Cada Emoção:")
        for emotion, prob in zip(model.classes_, probabilities[0]):
            st.write(f"{emotion.capitalize()}: {prob:.2f}")

    except Exception as e:
        st.error("Erro ao processar o arquivo de áudio. Verifique o formato e tente novamente.")
