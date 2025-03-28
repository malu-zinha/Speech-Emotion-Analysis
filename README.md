# Reconhecimento de Emoções em Áudio

Este é um projeto de **classificação de emoções em áudio**, onde é utilizado o dataset **RAVDESS** para treinar um modelo capaz de identificar a emoção presente em arquivos de áudio enviados pelos usuários. A aplicação conta com uma interface interativa desenvolvida em **Streamlit**, permitindo que os usuários enviem áudios para análise.

### 🎥 Vídeo Explicativo: [Mini Projeto 2 - Trilha](https://www.youtube.com/watch?v=xf8GMCGjloQ&ab_channel=LuigiSchmitt)

![Descrição da Imagem](https://i.imgur.com/33MqEWQ.png)

## Pipeline do Projeto

### 1. Coleta e Organização dos Dados

- **Dataset Utilizado**: [RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song)](https://zenodo.org/record/1188976)
  - O dataset contém gravações de áudios com diferentes emoções atuadas, como: alegria, tristeza, raiva, calma, medo, nojo, surpresa e neutro.
  - As emoções são representadas por números nos nomes dos arquivos.

- **Organização**:
  - Modality (01 = full-AV, 02 = video-only, 03 = audio-only).
  - Vocal channel (01 = speech, 02 = song).
  - Emotion (01 = neutral, 02 = calm, 03 = happy, 04 = sad, 05 = angry, 06 = fearful, 07 = disgust, 08 = surprised).
  - Emotional intensity (01 = normal, 02 = strong). NOTE: There is no strong intensity for the 'neutral' emotion.
  - Statement (01 = "Kids are talking by the door", 02 = "Dogs are sitting by the door").
  - Repetition (01 = 1st repetition, 02 = 2nd repetition).
  - Actor (01 to 24. Odd numbered actors are male, even numbered actors are female).

### 2. Extração de Features

A extração de features é uma etapa fundamental para converter os arquivos de áudio em representações numéricas compreensíveis pelo modelo de machine learning. Foram utilizadas as seguintes técnicas:

- **MFCCs (Mel-Frequency Cepstral Coefficients)**: Capturam as características espectrais do áudio.
- **Chroma Features**: Representam as energias das notas musicais.
- **Spectral Contrast**: Diferença entre os picos e vales no espectro de frequências.
- **Zero-Crossing Rate**: Taxa de mudança de sinal no áudio.

As features foram extraídas usando a biblioteca **librosa** e normalizadas para melhor desempenho do modelo.

### 3. Treinamento do Modelo

- **Modelo Utilizado**:
  - Um modelo de aprendizado de máquina foi treinado para classificar as emoções com base nas features extraídas.
  - Frameworks como **TensorFlow/Keras** foram utilizados.

- **Divisão dos Dados**:
  - O dataset foi dividido em conjuntos de treinamento, validação e teste.
  - Foi utilizada validação cruzada para evitar overfitting e avaliar a performance do modelo.

### 4. Interface com Streamlit

- **Funcionalidades**:
  - Upload de arquivos de áudio pelo usuário.
  - Predição da emoção baseada no modelo treinado.
  - Exibição de resultados com a probabilidade de cada emoção.

### 5. Avaliação do Modelo

- **Métricas de Avaliação**:
  - Acurácia
  - Matriz de confusão
  - F1-score, precisão e recall

- **Resultados**:
  - O desempenho do modelo foi avaliado nos dados de teste, e os resultados foram apresentados em forma de relatórios e gráficos no Streamlit.

---

## Como Executar o Projeto

1. Clone este repositório:
   ```bash
   git clone https://github.com/seu-usuario/audio-emotion-classification.git
   ```

2. Instale as dependências:
   ```bash
   pip install -r requirements.txt
   ```

3. Baixe o dataset RAVDESS e organize os arquivos conforme descrito no código.

4. Execute o aplicativo Streamlit:
   ```bash
   streamlit run app.py
   ```

5. Envie um arquivo de áudio no aplicativo e veja o modelo predizer a emoção presente.

---

## Tecnologias Utilizadas

- **Python**
- **Streamlit**
- **Librosa**
- **TensorFlow/Keras**
- **Matplotlib/Seaborn** (para visualizações)

---

## Próximos Passos

- Adicionar suporte para outros datasets e emoções.
- Melhorar a interface do Streamlit com gráficos interativos.
- Implementar detecção em tempo real para gravações ao vivo.

---

## Conclusão
Podemos ver que nosso modelo é mais preciso na predição das emoções surpresa e raiva, o que faz sentido, pois os arquivos de áudio dessas emoções diferem bastante dos outros em aspectos como tom, velocidade, etc. Caso você queira contribuir sinta-se a vontade e me contate em: schmittluigi@gmail.com ou se você me conhece fale comigo :)

---

