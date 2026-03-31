# ADNE - Proyecto: Análisis de Datos No Estructurados (2025-2026)

> **Dataset principal:** [ATCOSIM Corpus](https://huggingface.co/datasets/Jzuluaga/atcosim_corpus) — comunicaciones reales entre controladores aéreos y pilotos (~10h de audio con transcripciones manuales)

Este proyecto analiza comunicaciones de control de tráfico aéreo (ATC) desde dos ángulos complementarios: el audio en bruto y las transcripciones textuales. La idea es recorrer el pipeline completo del análisis de datos no estructurados, desde entender los datos hasta aplicar las técnicas más avanzadas de deep learning y transfer learning.

El corpus ATCOSIM es especialmente interesante porque es un dominio muy específico: vocabulario restringido, mucho ruido de fondo, acentos variados y una estructura de comunicación muy particular (call signs, instrucciones estandarizadas, readbacks...). Eso lo hace tanto un reto realista como un buen banco de pruebas.

---

## Estructura del Proyecto

```
├── AUDIO
│   ├── ADNE_Proyecto_Audio_ATC_EDA.ipynb               # Parte 1 — EDA
│   ├── ADNE_Proyecto_Audio_Part2_ML_DL.ipynb          # Parte 2 — ML Clásico + DL from scratch
│   └── ADNE_Proyecto_Audio_Part3_TransferLearning.ipynb  # Parte 3 — Transfer Learning + ASR
│
└── TEXTO
    ├── ADNE_Proyecto_Texto_Part4_NLP_Clasico.ipynb    # Parte 4 — NLP Clásico
    └── ADNE_Proyecto_Texto_Part5_DeepLearning.ipynb   # Parte 5 — Deep Learning + BERT
```

---

## Bloque AUDIO

### Notebook 1 — EDA: `ADNE_Proyecto_Audio_ATC_v2.ipynb`

Antes de tocar ningún modelo, hay que entender qué tenemos. Este notebook es el punto de partida de todo el proyecto.

Lo primero es cargar el dataset directamente desde Hugging Face y hacerse una idea general: cuántos audios hay, cuánto duran, qué transcripciones tienen, si las clases están balanceadas... Sin este paso es fácil entrenar modelos sobre datos mal entendidos.

Una vez claro el terreno, se extraen las features de audio que se usarán en los modelos posteriores: MFCCs (los más habituales en tareas de audio), centroide espectral, rolloff, bandwidth, ZCR y características de croma. Se incluyen también visualizaciones de formas de onda y espectrogramas para inspeccionar visualmente el corpus.

Como ATCOSIM no viene con etiquetas de categoría de instrucción, se crean manualmente a partir de palabras clave presentes en las transcripciones: `takeoff`, `landing`, `routing`, `readback`, `holding`, `frequency_change`. Esto permite convertir el problema en clasificación supervisada para los notebooks siguientes.

**Librerías principales:** `librosa`, `datasets`, `pandas`, `matplotlib`, `seaborn`

---

### Notebook 2 — ML Clásico + DL: `ADNE_Proyecto_Audio_Part2_ML_DL.ipynb`

Con las features del notebook anterior, aquí se compara qué tan lejos llegan los métodos clásicos frente a una red neuronal entrenada desde cero.

Se empieza con SVM (kernel RBF) y Random Forest porque son los dos modelos de referencia más razonables para features tabulares como los MFCCs: son rápidos, interpretables y generalmente dan buenos resultados con features bien construidas. Comparar ambos sirve también para ver qué tanto importa la linealidad del modelo en este dominio.

Después se implementa una CNN 1D en PyTorch (`CNN1D_ATC`). Se elige una CNN 1D y no una 2D porque las features espectrales tienen estructura temporal que conviene explotar en esa dimensión, pero no necesariamente en dos dimensiones como haría una CNN de imagen. Se añade early stopping para evitar overfitting y se visualizan las curvas de entrenamiento para diagnosticar el proceso.

**Librerías principales:** `scikit-learn`, `torch`, `librosa`, `tqdm`

---

### Notebook 3 — Transfer Learning + ASR: `ADNE_Proyecto_Audio_Part3_TransferLearning.ipynb`

Este notebook responde a una pregunta importante: ¿merece la pena usar modelos pre-entrenados en lugar de construir todo desde cero?

La respuesta, en el caso del audio, es casi siempre que sí. Se utiliza Wav2Vec 2.0 (`facebook/wav2vec2-base`) para extraer embeddings ricos del audio y se clasifican con un Random Forest encima. Comparar este resultado con el del notebook anterior da una idea clara del salto que supone el transfer learning incluso sin hacer fine-tuning.

Aparte de la clasificación, se aborda el reconocimiento automático de voz (ASR) con Whisper de OpenAI. Se transcriben los audios automáticamente y se evalúan con Word Error Rate (WER), que es la métrica estándar para ASR. Esto es relevante porque en la práctica muchas veces no se tiene acceso a transcripciones manuales y hay que generarlas.

Por último, se incluye un análisis teórico comparando APIs comerciales (Google, Azure, OpenAI) en términos de rendimiento y coste para procesar 1000h de audio, lo que contextualiza las decisiones de diseño en un escenario real.

Las transcripciones generadas aquí se exportan para alimentar el bloque de texto.

**Librerías principales:** `transformers`, `whisper`, `jiwer`, `torchaudio`, `datasets`

---

## Bloque TEXTO

### Notebook 4 — NLP Clásico: `ADNE_Proyecto_Texto_Part4_NLP_Clasico.ipynb`

Una vez tenemos texto (las transcripciones del corpus, ya sean manuales o generadas por Whisper), se aplica el pipeline estándar de NLP clásico. Este notebook sirve tanto como baseline de referencia como para entender bien el vocabulario y la estructura del corpus antes de usar modelos más pesados.

El preprocesamiento incluye limpieza, eliminación de stopwords y lematización: pasos necesarios en NLP clásico para reducir el ruido léxico. Con TF-IDF y Bag of Words se construyen representaciones vectoriales del texto; con Word2Vec se añade una representación semántica entrenada sobre el propio corpus, lo que captura relaciones específicas del dominio ATC que un modelo genérico no tendría.

Para explorar la estructura latente del corpus se aplican K-Means (clustering no supervisado) y LDA (topic modeling), que permiten ver si las categorías que definimos manualmente se corresponden con agrupaciones naturales en los datos.

Finalmente se comparan tres clasificadores: Naive Bayes, Regresión Logística y Random Forest, para tener una línea base sólida antes de pasar a deep learning.

**Librerías principales:** `nltk`, `gensim`, `scikit-learn`, `wordcloud`

---

### Notebook 5 — Deep Learning + BERT: `ADNE_Proyecto_Texto_Part5_DeepLearning.ipynb`

El último notebook cierra el proyecto con los modelos más potentes disponibles para clasificación de texto.

Primero se implementa una LSTM bidireccional (`LSTM_Classifier`) en PyTorch. Las LSTMs son el punto de entrada natural al deep learning secuencial: capturan dependencias de largo alcance en el texto mejor que los modelos clásicos, y la bidireccionalidad permite que el modelo tenga contexto tanto hacia adelante como hacia atrás al procesar cada token. Se añade un scheduler `ReduceLROnPlateau` para adaptar el learning rate durante el entrenamiento.

Después se hace fine-tuning de BERT (`bert-base-uncased`). La razón de usar BERT en lugar de entrenar un transformer desde cero es simple: el corpus ATC es pequeño y los transformers necesitan mucho más dato del que tenemos. Con fine-tuning aprovechamos el conocimiento lingüístico pre-entrenado y solo adaptamos las últimas capas al dominio específico.

El notebook termina con una tabla comparativa de todos los modelos del proyecto (Notebooks 4 y 5), exportada a `final_comparison_all_models.csv`, lo que permite ver de forma clara qué ha ganado cada paso en la progresión metodológica.

**Librerías principales:** `torch`, `transformers`, `datasets`, `scikit-learn`

---

## Dependencias entre Notebooks

```
Audio ATC v2 (EDA)
    └──> Audio Part2 (ML/DL)          ← usa las categorías definidas en EDA
    └──> Audio Part3 (TL/ASR)         ← usa las categorías + exporta transcripciones
                                               ↓
                                Texto Part4 (NLP Clásico)  ← puede usar transcripciones del Audio
                                               ↓
                                Texto Part5 (DL + BERT)    ← carga CSV del Notebook 4
```

---

## Requisitos

- Python 3.8+
- GPU recomendada para los notebooks de DL (compatible con Google Colab)
- Los notebooks de Audio detectan automáticamente si se ejecutan en Colab y ajustan la instalación

```bash
pip install pandas numpy matplotlib seaborn librosa soundfile datasets
pip install torch torchaudio transformers jiwer
pip install scikit-learn gensim wordcloud nltk
```

---

## Resumen de Modelos por Notebook

| Notebook | Modelos |
|----------|---------|
| Audio Part 2 | SVM, Random Forest, CNN 1D |
| Audio Part 3 | Wav2Vec 2.0 + RF, Whisper ASR |
| Texto Part 4 | Naive Bayes, Logistic Regression, Random Forest |
| Texto Part 5 | LSTM, BERT fine-tuned |
