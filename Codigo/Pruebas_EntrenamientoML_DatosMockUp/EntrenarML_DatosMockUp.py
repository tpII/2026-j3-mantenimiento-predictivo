import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras import layers, models


#GENERACIÓN DEL DATASET MOCK-UP (MPU6050)

fs = 100  # Frecuencia de muestreo: 100 Hz
duration_per_state = 60  # 60 segundos por estado
samples_per_state = fs * duration_per_state  # 6,000 muestras por estado

np.random.seed(42)
t = np.linspace(0, duration_per_state, samples_per_state, endpoint=False)

# Clase 0: Apagado (Solo ruido estático + Gravedad en Z = 1g)
ax0 = np.random.normal(0, 0.01, samples_per_state)
ay0 = np.random.normal(0, 0.01, samples_per_state)
az0 = 1.0 + np.random.normal(0, 0.01, samples_per_state)

# Clase 1: Normal Sin Carga (50 Hz fundamental, amplitud limpia)
ax1 = 0.15 * np.sin(2 * np.pi * 50 * t) + np.random.normal(0, 0.03, samples_per_state)
ay1 = 0.12 * np.cos(2 * np.pi * 50 * t) + np.random.normal(0, 0.03, samples_per_state)
az1 = 1.0 + 0.08 * np.sin(2 * np.pi * 100 * t) + np.random.normal(0, 0.03, samples_per_state)

# Clase 2: Normal Con Carga (45 Hz fundamental bajo demanda, amplitud moderada)
ax2 = 0.25 * np.sin(2 * np.pi * 45 * t) + np.random.normal(0, 0.04, samples_per_state)
ay2 = 0.20 * np.cos(2 * np.pi * 45 * t) + np.random.normal(0, 0.04, samples_per_state)
az2 = 1.0 + 0.15 * np.sin(2 * np.pi * 90 * t) + np.random.normal(0, 0.04, samples_per_state)

# Clase 3: Advertencia / Degradación (Desbalance 48 Hz + 2do Armónico 96 Hz)
ax3 = 0.55 * np.sin(2 * np.pi * 48 * t) + 0.30 * np.sin(2 * np.pi * 96 * t) + np.random.normal(0, 0.08, samples_per_state)
ay3 = 0.45 * np.cos(2 * np.pi * 48 * t) + 0.25 * np.cos(2 * np.pi * 96 * t) + np.random.normal(0, 0.08, samples_per_state)
az3 = 1.0 + 0.35 * np.sin(2 * np.pi * 96 * t) + np.random.normal(0, 0.08, samples_per_state)

# Clase 4: Anómala Crítica (Alta amplitud + Picos aleatorios de golpeteo)
spikes_x = (np.random.rand(samples_per_state) > 0.98) * np.random.choice([-1.5, 1.5], samples_per_state)
spikes_y = (np.random.rand(samples_per_state) > 0.98) * np.random.choice([-1.5, 1.5], samples_per_state)
ax4 = 1.20 * np.sin(2 * np.pi * 42 * t) + 0.70 * np.sin(2 * np.pi * 84 * t) + spikes_x + np.random.normal(0, 0.15, samples_per_state)
ay4 = 1.00 * np.cos(2 * np.pi * 42 * t) + 0.60 * np.cos(2 * np.pi * 84 * t) + spikes_y + np.random.normal(0, 0.15, samples_per_state)
az4 = 1.0 + 0.80 * np.sin(2 * np.pi * 84 * t) + np.random.normal(0, 0.15, samples_per_state)

# Consolidación del DataFrame
df = pd.DataFrame({
    'ax': np.concatenate([ax0, ax1, ax2, ax3, ax4]),
    'ay': np.concatenate([ay0, ay1, ay2, ay3, ay4]),
    'az': np.concatenate([az0, az1, az2, az3, az4]),
    'label': np.concatenate([np.full(samples_per_state, 0),
                            np.full(samples_per_state, 1),
                            np.full(samples_per_state, 2),
                            np.full(samples_per_state, 3),
                            np.full(samples_per_state, 4)])
})

# Guardar dataset simulado
df.to_csv('dataset_mockup_mpu6050.csv', index=False)
print("Dataset simulado guardado exitosamente como 'dataset_mockup_mpu6050.csv'.")


# 2. VENTANEO (1 segundo, 50% Overlap)

def create_windows(df, window_size=100, stride=50):
    X_list, y_list = [], []
    for label in df['label'].unique():
        sub_df = df[df['label'] == label].reset_index(drop=True)
        data = sub_df[['ax', 'ay', 'az']].values
        for start in range(0, len(data) - window_size + 1, stride):
            X_list.append(data[start:start + window_size])
            y_list.append(label)
    return np.array(X_list), np.array(y_list)

X, y = create_windows(df, window_size=100, stride=50)
print(f"Dimensiones de los tensores de entrada X: {X.shape} (Ventanas x Longitud x Canales)")

# División Entrenar / Validar (80% / 20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


# 3. ENTRENAMIENTO DE LA RED NEURONAL 1D-CNN

model = models.Sequential([
    layers.Input(shape=(100, 3)),
    layers.Conv1D(filters=16, kernel_size=3, activation='relu'),
    layers.MaxPooling1D(pool_size=2),
    layers.Conv1D(filters=32, kernel_size=3, activation='relu'),
    layers.MaxPooling1D(pool_size=2),
    layers.Flatten(),
    layers.Dense(32, activation='relu'),
    layers.Dense(5, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2, verbose=1)


# 4. EVALUACIÓN Y MATRIZ DE CONFUSIÓN

y_pred = np.argmax(model.predict(X_test), axis=1)
print("\n--- MATRIZ DE CONFUSIÓN ---")
print(confusion_matrix(y_test, y_pred))

print("\n--- REPORTE DE CLASIFICACIÓN ---")
target_names = ["0: Apagado", "1: Normal Sin Carga", "2: Normal Con Carga", "3: Advertencia", "4: Anómala Crítica"]
print(classification_report(y_test, y_pred, target_names=target_names))


# 5. CONVERSIÓN A TENSORFLOW LITE (.tflite)

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

with open('model_1dcnn.tflite', 'wb') as f:
    f.write(tflite_model)

print("Modelo optimizado y guardado como 'model_1dcnn.tflite' listo para integrar en el ESP32-S3.")