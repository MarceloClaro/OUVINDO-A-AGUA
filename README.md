# OUVINDO-A-AGUA -  Classificação de Sons de Água Vibrando em Copo de Vidro
Objetivo Geral Desenvolver um pipeline robusto e rigoroso, do ponto de vista técnico e científico, para classificar diferentes tipos de água (por exemplo: água mineral quente, água mineral gelada, água com sal, etc.) vibrando em um copo de vidro, a partir de sinais de áudio coletados.

---
 Classificação de Sons de Água Vibrando em Copo de Vidro

Este projeto implementa um pipeline completo para classificação de diferentes tipos de água vibrando em um copo de vidro, usando técnicas de Deep Learning (CNN) e Data Augmentation.

----------------------------------------
Instruções para Execução (via Streamlit)
----------------------------------------

1. Pré-requisitos:
   - Python 3.7+
   - Instalar as dependências:
     pip install streamlit librosa tensorflow seaborn audiomentations matplotlib scikit-learn

2. Preparação do Dataset:
   - Crie uma pasta chamada "dataset_agua".
   - Dentro dela, crie subpastas para cada classe (por exemplo: "AGUA MINERAL QUENTE", "AGUA MINERAL GELADA", etc.).
   - Em cada subpasta, coloque arquivos de áudio no formato .mp3, .m4a ou .wav.
   - Compacte a pasta "dataset_agua" em um arquivo ZIP.

3. Execução:
   - Salve o script `app.py` fornecido no mesmo diretório.
   - Rode: streamlit run app.py
   - No navegador, ao acessar o endereço exibido (geralmente http://localhost:8501), faça o upload do arquivo ZIP contendo o dataset_agua.
   - Clique em "Treinar Modelo" para iniciar o processo de extração de características, aplicação de Data Augmentation e treinamento da rede neural.
   - Após o treinamento, faça upload de um arquivo de áudio único para classificação.
   - Ajuste as caixas de seleção para ver diferentes visualizações (waveform, espectro de frequências, STFT, MFCC).

4. Estrutura do Código:
   - O código extrai MFCCs dos áudios e realiza Data Augmentation (ruído, pitch shift, time stretch e shift).
   - Uma CNN é treinada para classificação. Callbacks (EarlyStopping, ModelCheckpoint) são usados para evitar overfitting e salvar o melhor modelo.
   - Métricas (acurácia em treino/validação/teste, matriz de confusão e relatório de classificação) são exibidas.
   - O usuário pode visualizar a forma de onda, espectro de frequências (FFT), espectrograma STFT e MFCC do áudio de teste.

5. Notas sobre Desempenho:
   - Com um dataset muito pequeno, o modelo pode não se generalizar bem.  
   - Para melhores resultados, use mais dados, tente cross-validation, ajuste hiperparâmetros ou aplique outros classificadores.

----------------------------------------
Referências:
[1] Logan, B. (2000). Mel Frequency Cepstral Coefficients for Music Modeling. In ISMIR.

[2] Piczak, K.J. (2015). ESC: Dataset for Environmental Sound Classification. In ACM Multimedia.

[3] Salamon, J. & Bello, J.P. (2017). Deep Convolutional Neural Networks and Data Augmentation for Environmental Sound Classification. IEEE Signal Processing Letters.

----------------------------------------

