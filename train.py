import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import pandas as pd
import os
import cv2
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from utils import EMOTION_LABELS

def create_emotion_model(input_shape=(48, 48, 1), num_classes=7):
    model = Sequential([
        tf.keras.Input(shape=input_shape),
        Conv2D(32, (3, 3), activation='relu'),
        BatchNormalization(),
        Conv2D(32, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Dropout(0.25),

        Conv2D(64, (3, 3), activation='relu'),
        BatchNormalization(),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Dropout(0.25),

        Conv2D(128, (3, 3), activation='relu'),
        BatchNormalization(),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Dropout(0.25),

        Conv2D(256, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),

        Flatten(),
        Dense(512, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def train_model(model, train_generator, val_generator, epochs=50, model_save_path="models/emotion_model.keras"):
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    callbacks = [
        EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7, verbose=1),
        ModelCheckpoint(model_save_path, monitor='val_accuracy', save_best_only=True, verbose=1)
    ]

    print("Starting model training...")
    history = model.fit(train_generator,
                        epochs=epochs,
                        validation_data=val_generator,
                        callbacks=callbacks,
                        verbose=1)

    print(f"Model training completed. Best model saved to {model_save_path}")
    return history

def main():
    print("Starting Emotion Recognition Model Training...")
    data_path = "data/fer2013.csv"
    if os.path.exists(data_path):
        print("Loading FER2013 dataset...")
        X_train, X_val, X_test, y_train, y_val, y_test = load_fer2013_data(data_path)
    else:
        print("FER2013 dataset not found. Creating sample data...")
        X_train, X_val, X_test, y_train, y_val, y_test = create_sample_data(2000)

    if X_train is None:
        print("Failed to load data. Exiting...")
        return

    model = create_emotion_model()
    train_generator, val_generator = create_data_generators(X_train, y_train, X_val, y_val)
    history = train_model(model, train_generator, val_generator)
    plot_training_history(history)
    test_accuracy, predictions = evaluate_model(model, X_test, y_test)
    print("Training complete. Accuracy:", test_accuracy)

if __name__ == "__main__":
    np.random.seed(42)
    tf.random.set_seed(42)
    main()
