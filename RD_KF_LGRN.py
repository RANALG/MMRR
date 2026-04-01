# 2.1 Retornos t y etiqueta: si el retorno promedio en t+1 es positivo

X = df.drop('Date', axis=1).values[:-1]  
y_raw = df.drop('Date', axis=1).mean(axis=1).shift(-1) 
y = (y_raw > 0).astype(int).values[:-1]  # 1 if >0, else 0

# 2.2 One-hot encoding de y para categorical_crossentropy
from tensorflow.keras.utils import to_categorical
y_cat = to_categorical(y, num_classes=2)

# 2.3 Dividir en entrenamiento y prueba
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_cat, test_size=0.2, random_state=42, stratify=y
)

# 2.4 Escalar los datos
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
# 3.1 Arquitectura (input shape = 10)
from tensorflow import keras
from tensorflow.keras import layers

modelo = keras.Sequential([
    layers.Input(shape=(10,)),
    layers.Dense(15, activation='relu'),
    layers.Dense(2, activation='softmax')
])

modelo.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# 3.2 Callbacks para detener temprano y ajustar LR
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

early_stop = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10)


# 3.3 Entrenamiento
print("Entrenando el modelo…")
history = modelo.fit(
    X_train_scaled, y_train, 
    validation_split=0.2,
    epochs=25,
    verbose=1,
    callbacks=[early_stop, reduce_lr]
)
print("Modelo entrenado")

# 3.4 Evaluación final
loss, acc = modelo.evaluate(X_test_scaled, y_test, verbose=0) 
print(f"Test Loss: {loss:.4f} – Test Accuracy: {acc:.4f}")

# GRAFICA DE COMPORTAMIENTO
plt.figure(figsize=(8, 5))
plt.plot(history.history["loss"], label="train_loss")
plt.plot(history.history["val_loss"], label="val_loss")
plt.xlabel("Num de época")
plt.ylabel("Magnitud de pérdida")
plt.title("Comportamiento de la pérdida durante el entrenamiento")
plt.legend()
plt.show()