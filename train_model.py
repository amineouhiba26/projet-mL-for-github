import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras import layers, models, optimizers, callbacks, regularizers
from sklearn.utils.class_weight import compute_class_weight

# Configuration
MODELS_DIR = "models"
DATASET_DIR = "dataset/"
BATCH_SIZE = 16
EPOCHS = 15
IMAGE_SIZE = (224, 224)

os.makedirs(MODELS_DIR, exist_ok=True)

def train_model():
    print("[INFO] Loading dataset...")

    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        DATASET_DIR,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode='categorical',
        shuffle=True,
        seed=123,
        validation_split=0.2,
        subset='training'
    )

    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        DATASET_DIR,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode='categorical',
        shuffle=True,
        seed=123,
        validation_split=0.2,
        subset='validation'
    )

    class_names = train_ds.class_names
    num_classes = len(class_names)

    # Compute class weights
    labels = np.concatenate([y.numpy() for x, y in train_ds], axis=0)
    class_indices = np.argmax(labels, axis=1)
    class_weights = compute_class_weight('balanced', classes=np.unique(class_indices), y=class_indices)
    class_weight_dict = dict(enumerate(class_weights))

    # Model
    base_model = EfficientNetB0(
        input_shape=(*IMAGE_SIZE, 3),
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = False

    inputs = layers.Input(shape=(*IMAGE_SIZE, 3))
    x = layers.Rescaling(1./255)(inputs)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.001))(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(inputs, outputs)

    model.compile(
        optimizer=optimizers.Adam(learning_rate=1e-4),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # Callbacks
    callback_list = [
        callbacks.EarlyStopping(patience=3, restore_best_weights=True),
        callbacks.ModelCheckpoint(os.path.join(MODELS_DIR, "best_model.keras"), save_best_only=True),
        callbacks.ReduceLROnPlateau(factor=0.2, patience=2)
    ]

    print("[INFO] Starting training...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=150,
        class_weight=class_weight_dict,
        callbacks=callback_list
    )

    print("[INFO] Saving models...")
    model.save(os.path.join(MODELS_DIR, "model.keras"))

    # Convert to TFLite
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()

    with open(os.path.join(MODELS_DIR, "model.tflite"), 'wb') as f:
        f.write(tflite_model)

    print("[INFO] Model conversion complete!")

if __name__ == "__main__":
    train_model()
