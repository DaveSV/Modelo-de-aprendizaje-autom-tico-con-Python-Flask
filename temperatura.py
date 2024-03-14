import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Generar datos ficticios
np.random.seed(0) # Fijar la semilla para reproducibilidad
temperaturas = np.random.randint(20, 35, 100) # Temperaturas entre 20 y 35 grados Celsius
# Las ventas son una función lineal de las temperaturas más algo de ruido
ventas = 50 + (temperaturas - 20) * 3 + np.random.normal(0, 10, 100)

# Visualizar los datos generados
plt.scatter(temperaturas, ventas)
plt.title('Ventas de Conos de Helado vs. Temperatura')
plt.xlabel('Temperatura (°C)')
plt.ylabel('Ventas de Conos de Helado')
plt.show()

# Preparar los datos para la regresión lineal
temperaturas = temperaturas.reshape(-1, 1) # Formato requerido por scikit-learn
ventas = ventas.reshape(-1, 1)

# Dividir los datos en conjunto de entrenamiento y de prueba
X_train, X_test, y_train, y_test = train_test_split(temperaturas, ventas, test_size=0.2, random_state=0)

# Crear y entrenar el modelo de regresión lineal
modelo = LinearRegression()
modelo.fit(X_train, y_train) # El entrenamiento ocurre aquí

# Durante el entrenamiento, LinearRegression ajusta los parámetros (coeficientes) del modelo
# para minimizar el Error Cuadrático Medio (ECM) entre las ventas predichas y las reales.
# Esto se logra mediante un proceso de optimización.

# Predecir las ventas en el conjunto de prueba
predicciones = modelo.predict(X_test)

# Visualizar las predicciones y los datos de entrenamiento/prueba
plt.scatter(X_train, y_train, color='blue', label='Datos de Entrenamiento')
plt.scatter(X_test, y_test, color='green', label='Datos de Prueba')
plt.plot(X_test, predicciones, color='red', label='Línea de Regresión')
plt.title('Regresión Lineal: Ventas de Conos de Helado vs. Temperatura')
plt.xlabel('Temperatura (°C)')
plt.ylabel('Ventas de Conos de Helado')
plt.legend()
plt.show()

# Imprimir los parámetros del modelo
print(f'Pendiente (m): {modelo.coef_[0][0]}')
print(f'Intercepto (b): {modelo.intercept_[0]}')