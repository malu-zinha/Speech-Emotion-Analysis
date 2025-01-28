# Reconhecimento de Emoções em Áudio

Este é um projeto de **classificação de emoções em áudio**, onde é utilizado o dataset **RAVDESS** para treinar um modelo capaz de identificar a emoção presente em arquivos de áudio enviados pelos usuários. A aplicação conta com uma interface interativa desenvolvida em **Streamlit**, permitindo que os usuários enviem áudios para análise.

## Pipeline do Projeto

### 1. Coleta e Organização dos Dados

- **Dataset Utilizado**: [RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song)](https://zenodo.org/record/1188976)
  - O dataset contém gravações de áudios com diferentes emoções atuadas, como: alegria, tristeza, raiva, calma, medo, nojo, surpresa e neutro.
  - As emoções são representadas por números nos nomes dos arquivos.

- **Organização**:
  - Os arquivos de áudio foram carregados e classificados com base na emoção correspondente.
  - Um dataframe foi criado para armazenar as features extraídas e as emoções rotuladas.

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
  - Frameworks como **Scikit-learn** ou **TensorFlow/Keras** foram utilizados.

- **Divisão dos Dados**:
  - O dataset foi dividido em conjuntos de treinamento, validação e teste.
  - Foi utilizada validação cruzada para evitar overfitting e avaliar a performance do modelo.

### 4. Interface com Streamlit

- **Funcionalidades**:
  - Upload de arquivos de áudio pelo usuário.
  - Visualização do waveform e espectrograma do áudio enviado.
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
- **Scikit-learn** / **TensorFlow/Keras**
- **Matplotlib/Seaborn** (para visualizações)

---

## Próximos Passos

- Adicionar suporte para outros datasets e emoções.
- Melhorar a interface do Streamlit com gráficos interativos.
- Implementar detecção em tempo real para gravações ao vivo.

---

## Contribuições

Contribuições são bem-vindas! Sinta-se à vontade para abrir issues ou enviar pull requests com melhorias para este projeto.

---

**Autor:** [Seu Nome]  
**Contato:** seuemail@dominio.com

