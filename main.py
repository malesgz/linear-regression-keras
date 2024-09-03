# Importación de las dependecias.
import pandas as pd
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import Input
from tensorflow.keras.optimizers import SGD
import matplotlib.pyplot as plt

# Se leen los datos del archivo csv.
data = pd.read_csv('altura_peso.csv')
x = data['Altura'].values
y = data['Peso'].values

# Utilización del Modelo en Keras.
model = Sequential([
    Input(shape=(1,)),
    Dense(1, activation='linear')
])
optimizer = SGD(learning_rate=0.0004)
model.compile(optimizer=optimizer, loss='mse')

# Se entrena el módelo.
history = model.fit(x, y, epochs=10000, batch_size=len(x), verbose=0)

# Visualización de los datos, obtenemos y mostramos los parámetros w y b.
w, b = model.layers[0].get_weights()
print(f"w: {w[0][0]}, b: {b[0]}")

# Grafica de la pérdida durante las épocas.
plt.figure(figsize=(14, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'])
plt.title('MSE vs Épocas')
plt.xlabel('Épocas')
plt.ylabel('MSE')

# Gráfico de la recta de regresión de los datos.
plt.subplot(1, 2, 2)
plt.scatter(x, y, color='blue', label='Datos originales')
plt.plot(x, model.predict(x), color='red', label='Recta de regresión')
plt.title('Regresión Lineal con Keras')
plt.xlabel('Altura')
plt.ylabel('Peso')
plt.legend()
plt.show()

# Predicción.
altura = 176
altura_np = np.array([[altura]])
peso_predicho = model.predict([altura_np])
print(f"El peso predicho para una altura de {altura} cm es {peso_predicho[0][0]:.2f} kg")