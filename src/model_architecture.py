import tensorflow as tf
from keras import layers, models
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CNN3DModel:
    def __init__(self, input_shape, num_classes=2):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = None

    def build_model(self):
        model = models.Sequential([
            layers.Input(shape=self.input_shape),
            layers.Conv3D(8, (3, 3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling3D((2, 2, 2)),
            layers.Dropout(0.2),
            layers.Conv3D(16, (3, 3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling3D((2, 2, 2)),
            layers.Dropout(0.3),
            layers.Conv3D(32, (3, 3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.GlobalAveragePooling3D(),
            layers.Dropout(0.4),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        self.model = model
        return model

    def compile_model(self, learning_rate=0.001):
        if self.model is None:
            raise ValueError("Call build_model() first.")
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        logger.info("Model compiled.")
        self.model.summary()

    def get_callbacks(self, patience=15):
        return [
            tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True, verbose=1),
            tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=patience//2, min_lr=1e-7, verbose=1),
            tf.keras.callbacks.ModelCheckpoint('best_model.keras', monitor='val_accuracy', save_best_only=True, verbose=1)
        ]

    def train(self, X_train, y_train, X_val, y_val, epochs=100, batch_size=8):
        if self.model is None:
            raise ValueError("Call build_model() first.")
        callbacks = self.get_callbacks()
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        return history

    def evaluate(self, X_test, y_test):
        if self.model is None:
            raise ValueError("Model not built yet.")
        return self.model.evaluate(X_test, y_test, verbose=1)

