# app.py
# -*- coding: utf-8 -*-
"""
Classificação de Sons de Água Vibrando em Copo de Vidro (Apenas WAV)

Fluxo:
1. O usuário faz upload do arquivo ZIP contendo `dataset_agua` em subpastas com arquivos .wav.
2. O código extrai o dataset e treina um modelo CNN com Data Augmentation.
3. Avalia o modelo, mostra matriz de confusão e relatório de classificação.
4. Permite ao usuário fazer upload de um arquivo de áudio .wav para classificação,
   gerando visualizações (waveform, FFT, STFT, MFCC).

Observação:  
Apenas .wav é suportado para evitar dependências externas e problemas de backend.
"""

import os
import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import librosa
import librosa.display as ld
import seaborn as sns
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift, Shift
import zipfile
import streamlit as st
from io import BytesIO
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

sns.set()

# ==================== CONTROLE DE REPRODUTIBILIDADE ====================
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

st.title("Classificação de Sons de Água Vibrando em Copo de Vidro (Apenas WAV)")

st.write("1. Faça upload do arquivo `.zip` contendo a pasta `dataset_agua` com as classes de áudio em formato `.wav`.")
dataset_zip = st.file_uploader("Upload do dataset (zip):", type="zip")

temp_dir = "./temp_dataset"
if not os.path.exists(temp_dir):
    os.makedirs(temp_dir, exist_ok=True)

augment = Compose([
    AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5),
    TimeStretch(min_rate=0.8, max_rate=1.25, p=0.5),
    PitchShift(min_semitones=-4, max_semitones=4, p=0.5),
    Shift(min_shift=-0.5, max_shift=0.5, p=0.5),
])

def load_audio_with_augmentation(file_path, sr=None, apply_augmentation=True):
    # Apenas .wav
    data, sr = librosa.load(file_path, sr=sr, res_type='kaiser_fast')
    if apply_augmentation:
        data = augment(samples=data, sample_rate=sr)
    return data, sr

def extract_features(data, sr):
    mfccs = librosa.feature.mfcc(y=data, sr=sr, n_mfcc=40)
    mfccs_scaled = np.mean(mfccs.T, axis=0)
    return mfccs_scaled

if dataset_zip is not None:
    with zipfile.ZipFile(dataset_zip, 'r') as zip_ref:
        zip_ref.extractall(temp_dir)

    base_path = os.path.join(temp_dir, "dataset_agua")
    if not os.path.exists(base_path):
        st.error("O arquivo zip deve conter uma pasta `dataset_agua` na raiz.")
        st.stop()

    categories = os.listdir(base_path)
    categories = [c for c in categories if os.path.isdir(os.path.join(base_path, c))]

    if len(categories) == 0:
        st.error("Não foram encontradas classes dentro de `dataset_agua`. Verifique a estrutura.")
        st.stop()

    st.write("Classes encontradas:", categories)

    file_paths = []
    labels = []
    for cat in categories:
        cat_path = os.path.join(base_path, cat)
        # Apenas arquivos .wav
        files_in_cat = [f for f in os.listdir(cat_path) if f.lower().endswith('.wav')]
        if len(files_in_cat) == 0:
            st.warning(f"Nenhum arquivo .wav encontrado na classe {cat}. Por favor, converta seus áudios para .wav.")
        for file_name in files_in_cat:
            file_paths.append(os.path.join(cat_path, file_name))
            labels.append(cat)

    df = pd.DataFrame({'file_path': file_paths, 'class': labels})
    st.write("Total de amostras originais:", len(df))
    st.write(df.head())

    if len(df) == 0:
        st.error("Não há amostras para treinamento. Verifique seu dataset ou converta seus áudios para .wav.")
        st.stop()

    augment_factor = 2
    apply_augmentation = True

    if st.button("Treinar Modelo"):
        with st.spinner("Extraindo features e aplicando Data Augmentation..."):
            extracted_features = []
            final_labels = []
            try:
                for i in range(len(df)):
                    file = df['file_path'].iloc[i]
                    label = df['class'].iloc[i]
                    data, sr = load_audio_with_augmentation(file, sr=None, apply_augmentation=False)
                    original_feature = extract_features(data, sr)
                    extracted_features.append(original_feature)
                    final_labels.append(label)

                    if apply_augmentation:
                        for _ in range(augment_factor):
                            aug_data, sr = load_audio_with_augmentation(file, sr=None, apply_augmentation=True)
                            aug_feature = extract_features(aug_data, sr)
                            extracted_features.append(aug_feature)
                            final_labels.append(label)
            except Exception as e:
                st.error(f"Erro ao carregar o áudio. Certifique-se de usar .wav. Detalhes: {e}")
                st.stop()

        X = np.array(extracted_features)
        y = np.array(final_labels)

        st.write("Total de amostras após Data Augmentation:", len(X))

        labelencoder = LabelEncoder()
        y_encoded = to_categorical(labelencoder.fit_transform(y))

        X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=SEED)
        X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.5, random_state=SEED)

        X_train = X_train[:,:,np.newaxis]
        X_test = X_test[:,:,np.newaxis]
        X_val = X_val[:,:,np.newaxis]

        st.write("X_train shape:", X_train.shape)
        st.write("X_val shape:", X_val.shape)
        st.write("X_test shape:", X_test.shape)
        st.write("Número de classes:", y_encoded.shape[1])

        model = Sequential()
        model.add(Conv1D(64, kernel_size=10, activation='relu', input_shape=(X_train.shape[1], 1)))
        model.add(Dropout(0.4))
        model.add(MaxPooling1D(pool_size=4))
        model.add(Conv1D(128, kernel_size=10, padding='same', activation='relu'))
        model.add(Dropout(0.4))
        model.add(MaxPooling1D(pool_size=4))
        model.add(Flatten())
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.4))
        model.add(Dense(y_encoded.shape[1], activation='softmax'))

        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        num_epochs = 40
        batch_size = 32

        checkpointer = ModelCheckpoint(filepath='saved_models/model_agua_augmented.keras', verbose=1, save_best_only=True)
        earlystop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

        with st.spinner("Treinando o modelo..."):
            history = model.fit(X_train, y_train,
                                epochs=num_epochs,
                                batch_size=batch_size,
                                validation_data=(X_val, y_val),
                                callbacks=[checkpointer, earlystop],
                                verbose=1)
        st.success("Treinamento concluído!")

        score_train = model.evaluate(X_train, y_train, verbose=0)
        score_val = model.evaluate(X_val, y_val, verbose=0)
        score_test = model.evaluate(X_test, y_test, verbose=0)

        st.write("Acurácia Treino: {:.2f}%".format(score_train[1]*100))
        st.write("Acurácia Validação: {:.2f}%".format(score_val[1]*100))
        st.write("Acurácia Teste: {:.2f}%".format(score_test[1]*100))

        y_pred = model.predict(X_test)
        y_pred_classes = y_pred.argmax(axis=1)
        y_true = y_test.argmax(axis=1)

        all_classes = range(len(labelencoder.classes_))
        cm = confusion_matrix(y_true, y_pred_classes, labels=all_classes)

        cm_df = pd.DataFrame(cm, index=labelencoder.classes_, columns=labelencoder.classes_)

        fig, ax = plt.subplots(figsize=(12,8))
        sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_title("Matriz de Confusão")
        ax.set_xlabel("Classe Prevista")
        ax.set_ylabel("Classe Real")
        st.pyplot(fig)

        st.write("Relatório de Classificação:")
        report = classification_report(y_true, y_pred_classes, labels=all_classes,
                                      target_names=labelencoder.classes_, zero_division=0)
        st.text(report)

        st.session_state.model = model
        st.session_state.labelencoder = labelencoder
        st.session_state.sr = sr

else:
    st.info("Após enviar o dataset .zip, clique em 'Treinar Modelo' para iniciar o processo.")

st.header("Classificar Novo Áudio (somente .wav)")

uploaded_file = st.file_uploader("Envie um arquivo de áudio .wav para classificar:", type=["wav"])

plot_waveform_flag = st.checkbox("Mostrar Waveform", value=True)
plot_frequency_flag = st.checkbox("Mostrar Espectro de Frequências (FFT)", value=True)
plot_spectrogram_flag = st.checkbox("Mostrar Spectrograma STFT", value=True)
plot_mfcc_flag = st.checkbox("Mostrar Espectrograma MFCC", value=True)

if uploaded_file is not None and 'model' in st.session_state and 'labelencoder' in st.session_state:
    model = st.session_state.model
    labelencoder = st.session_state.labelencoder

    try:
        data, sr = librosa.load(uploaded_file, sr=None, res_type='kaiser_fast')
    except Exception as e:
        st.error(f"Erro ao carregar o áudio. Use um arquivo .wav. Detalhes: {e}")
        st.stop()

    mfccs = librosa.feature.mfcc(y=data, sr=sr, n_mfcc=40)
    mfccs_scaled = np.mean(mfccs.T, axis=0)
    mfccs_scaled = mfccs_scaled.reshape(1,-1)
    mfccs_scaled = mfccs_scaled[:,:,np.newaxis]

    prediction = model.predict(mfccs_scaled)
    pred_class = prediction.argmax(axis=1)
    pred_label = labelencoder.inverse_transform(pred_class)

    st.write("Classificação/resultado:", pred_label[0])

    def plot_waveform_func(data, sr, title="Waveform"):
        fig, ax = plt.subplots(figsize=(14,4))
        ax.set_title(title)
        ld.waveshow(data, sr=sr, ax=ax)
        ax.set_xlabel("Tempo (s)")
        ax.set_ylabel("Amplitude")
        return fig

    def plot_frequency_spectrum_func(data, sr, title="Espectro de Frequências"):
        N = len(data)
        fft = np.fft.fft(data)
        fft = np.abs(fft[:N//2])
        freqs = np.fft.fftfreq(N, 1/sr)[:N//2]
        fig, ax = plt.subplots(figsize=(14,4))
        ax.set_title(title)
        ax.plot(freqs, fft)
        ax.set_xlabel("Frequência (Hz)")
        ax.set_ylabel("Amplitude")
        ax.grid(True)
        return fig

    def plot_spectrogram_func(data, sr, title="Spectrograma (STFT)"):
        D = np.abs(librosa.stft(data))
        DB = librosa.amplitude_to_db(D, ref=np.max)
        fig, ax = plt.subplots(figsize=(14,4))
        ax.set_title(title)
        mappable = ld.specshow(DB, sr=sr, x_axis='time', y_axis='hz', cmap='magma', ax=ax)
        plt.colorbar(mappable=mappable, ax=ax, format='%+2.0f dB')
        ax.set_xlabel("Tempo (s)")
        ax.set_ylabel("Frequência (Hz)")
        return fig

    def plot_mfcc_func(data, sr, title="Espectrograma (MFCC)"):
        mfccs = librosa.feature.mfcc(y=data, sr=sr, n_mfcc=40)
        mfccs_db = librosa.amplitude_to_db(np.abs(mfccs))
        fig, ax = plt.subplots(figsize=(14,4))
        ax.set_title(title)
        mappable = ld.specshow(mfccs_db, x_axis='time', y_axis='log', cmap='Spectral', sr=sr, ax=ax)
        plt.colorbar(mappable=mappable, ax=ax, format='%+2.f dB')
        return fig

    if plot_waveform_flag:
        fig = plot_waveform_func(data, sr, title=f"Waveform - {pred_label[0]}")
        st.pyplot(fig)

    if plot_frequency_flag:
        fig = plot_frequency_spectrum_func(data, sr, title=f"Espectro de Frequências - {pred_label[0]}")
        st.pyplot(fig)

    if plot_spectrogram_flag:
        fig = plot_spectrogram_func(data, sr, title=f"Spectrograma STFT - {pred_label[0]}")
        st.pyplot(fig)

    if plot_mfcc_flag:
        fig = plot_mfcc_func(data, sr, title=f"Espectrograma MFCC - {pred_label[0]}")
        st.pyplot(fig)

elif uploaded_file is not None:
    st.warning("O modelo ainda não foi treinado. Treine o modelo antes de fazer previsões.")

st.write("Para melhor desempenho, considere usar datasets maiores, validação cruzada e ajuste de hiperparâmetros.")
