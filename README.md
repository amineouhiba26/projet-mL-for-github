# 🔌 Accessory Classifier

Un projet de classification d’images d’accessoires électroniques (chargeur, clavier, smartphone, etc.) basé sur l’apprentissage profond, utilisant **TensorFlow**, **EfficientNet**, et une interface web avec **Streamlit**.

---

## 📁 Structure du projet

```
.
├── dataset/                  # Dataset d’entraînement (1 dossier par classe)
├── models/
│   ├── model.keras           # Modèle Keras complet
│   └── model.tflite          # Version TFLite optimisée
├── train_model.py            # Script d’entraînement
├── predict_model.py          # Script de prédiction
├── app.py                    # Interface Streamlit
├── requirements.txt          # Dépendances Python
└── README.md                 # Documentation
```

---

## 🚀 Lancer le projet

### 1. Installer les dépendances

```bash
pip install -r requirements.txt
```

### 2. Préparer les données

Structure du dossier `dataset/` :

```
dataset/
├── Charger/
│   └── image1.jpg
├── Headphone/
│   └── image2.jpg
...
```

Chaque sous-dossier représente une **classe**.

---

## 🧠 Entraînement du modèle

Lancer l'entraînement :

```bash
python train_model.py
```

Ce script :

- Charge les images depuis `dataset/`
- Sépare en train/validation (80%/20%)
- Applique **EfficientNetB0** (transfer learning)
- Ajoute :
  - `GlobalAveragePooling2D`
  - `Dense(128, relu)` + Dropout
  - `Dense(num_classes, softmax)`
- Utilise des **class weights** pour équilibrer les classes
- Sauvegarde deux modèles :
  - `models/model.keras`
  - `models/model.tflite` (optimisé pour Streamlit)

---

## 🧪 Prédiction (avec Streamlit)

Lancer l’interface :

```bash
streamlit run app.py
```

Dans le navigateur :
- Uploade une image
- Le modèle TFLite est chargé
- L’image est prétraitée : redimensionnée, normalisée
- Prédiction avec `interpreter.invoke()`
- Affichage :
  - Classe prédite
  - Confiance (score softmax)

---

## 🧾 Détails du code

### `train_model.py`

- Utilise `tf.keras.preprocessing.image_dataset_from_directory` pour charger les données
- Base : `EfficientNetB0(weights='imagenet', include_top=False)`
- Tête personnalisée avec `Dense`, `Dropout`, `GlobalAveragePooling2D`
- Optimisation :
  - `Adam`, `EarlyStopping`, `ReduceLROnPlateau`, `ModelCheckpoint`
- Conversion finale vers `.tflite`

### `predict_model.py`

- Lit le fichier image avec `PIL.Image`
- Redimensionne à `224x224`, applique le prétraitement d’EfficientNet
- Charge le modèle `.tflite` avec `tf.lite.Interpreter`
- Donne une prédiction softmax + argmax

### `app.py`

- Interface web simple avec `streamlit`
- Appel de `predict_accessory(uploaded_file)` pour analyser l’image
- Affichage des résultats avec `st.image`, `st.success`, `st.info`

---

## ✅ Requirements

- Python ≥ 3.7
- TensorFlow ≥ 2.10
- Streamlit
- NumPy
- Pillow
- scikit-learn

---

## ✍️ Auteur

**Amine Ouhiba**  
GitHub: [@amineouhiba26](https://github.com/amineouhiba26)