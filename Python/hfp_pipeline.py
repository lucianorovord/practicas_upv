import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score


# =============================================================================
# 1. EDA — Exploratory Data Analysis
# =============================================================================
# Antes de construir cualquier modelo, hay que entender los datos:
# cuántas filas/columnas hay, qué tipos tienen, si hay valores nulos, etc.
# Este paso nunca modifica los datos; solo los observa.
#
#heart = pd.read_csv('./heart.csv')   # ajusta la ruta si es necesario
#
#print("=== Forma del dataset ===")
#print(heart.shape)            # (918, 12)
#
#print("\n=== Tipos de datos y nulos ===")
#print(heart.info())
#
#print("\n=== Estadísticas descriptivas ===")
#print(heart.describe())
#
## Comprobamos cuántos valores "imposibles" hay (0 en columnas médicas)
#print(f"\nCholesterol == 0 : {len(heart[heart['Cholesterol'] == 0])} filas")
#print(f"RestingBP  == 0 : {len(heart[heart['RestingBP']  == 0])} filas")
#
#
## =============================================================================
# 2. PREPROCESAMIENTO
# =============================================================================
# El preprocesamiento transforma los datos en bruto en algo que la red
# neuronal puede aprender. Tiene tres subfases:
#   a) Fixing problems  → arreglar valores incorrectos
#   b) Encoding         → convertir categorías a números
#   c) Scaling          → llevar todas las columnas al mismo rango


#--- 2a. Fixing problems ---
# Un colesterol de 0 mg/dL es fisiológicamente imposible; igual que una
# presión arterial de 0 mmHg. Los sustituimos por la media del resto de
# filas válidas para no perder esos registros.
#
#mean_chol = heart[heart['Cholesterol'] != 0]['Cholesterol'].mean()
#mean_bp   = heart[heart['RestingBP']   != 0]['RestingBP'].mean()
#
#heart.loc[heart['Cholesterol'] == 0, 'Cholesterol'] = round(mean_chol)
#heart.loc[heart['RestingBP']   == 0, 'RestingBP']   = round(mean_bp)
#
#print(f"\nMedia Cholesterol (sin ceros): {mean_chol:.1f}")
#print(f"Media RestingBP   (sin ceros): {mean_bp:.1f}")
#
#
## --- 2b. Encoding ---
## Las redes neuronales solo operan con números. Las columnas con texto
## ('M'/'F', 'ATA'/'ASY', etc.) se mapean a enteros mediante un diccionario.
## Nota: este "label encoding" es suficiente para una MLP; si usáramos un
## árbol de decisión u otro modelo sensible al orden, consideraríamos
## one-hot encoding para las variables nominales.
#
#heart['Sex']           = heart['Sex'].map({'M': 1, 'F': 0}).astype(int)
#heart['ChestPainType'] = heart['ChestPainType'].map({'ATA': 0, 'NAP': 1, 'ASY': 2, 'TA': 3}).astype(int)
#heart['RestingECG']    = heart['RestingECG'].map({'Normal': 0, 'ST': 1, 'LVH': 2}).astype(int)
#heart['ExerciseAngina']= heart['ExerciseAngina'].map({'N': 0, 'Y': 1}).astype(int)
#heart['ST_Slope']      = heart['ST_Slope'].map({'Flat': 0, 'Up': 1, 'Down': 2}).astype(int)
#
#print("\n=== Primeras filas tras encoding ===")
#print(heart.head())
#
#
## =============================================================================
## 3. SPLIT 70 / 15 / 15
## =============================================================================
## Dividimos los datos en tres conjuntos con roles distintos:
##
##   TRAIN  (70%) → la red aprende ajustando sus pesos con estos ejemplos
##   VAL    (15%) → durante el entrenamiento, mide si generaliza bien
##                  (permite detectar overfitting sin contaminar el test)
##   TEST   (15%) → evaluación final, única vez; simula datos "del mundo real"
##
## La separación entre val y test es clave: si usamos el test para tomar
## decisiones (elegir hiperparámetros), estamos "haciendo trampa" porque el
## modelo ha visto indirectamente esos datos.
#
#y = heart['HeartDisease']
#X = heart.drop(columns=['HeartDisease'])
#
## Paso 1: separar train (70%) del resto (30%)
#X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, random_state=42)
#
## Paso 2: dividir ese 30% en val (15%) y test (15%)
#X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42)
#
#print(f"\nX_train: {X_train.shape}  |  X_val: {X_val.shape}  |  X_test: {X_test.shape}")
#
#
## =============================================================================
## 4. SCALING — StandardScaler
## =============================================================================
## Las columnas tienen rangos muy distintos: Age va de 28 a 77, MaxHR de 60
## a 202. Si no escalamos, las neuronas darán más "peso visual" a las columnas
## con valores grandes, aunque no sean las más informativas.
##
## StandardScaler estandariza cada columna: substrae la media y divide por la
## desviación estándar → resultado: media ≈ 0, std ≈ 1.
##
## REGLA DE ORO:
##   - fit_transform en TRAIN: aprende la media y std del train
##   - transform en VAL y TEST: aplica esa MISMA escala (no aprende de nuevo)
##
## Si hiciéramos fit en val o test, estaríamos usando información del futuro
## ("data leakage"), lo que produce estimaciones demasiado optimistas.
#
#scaler = StandardScaler()
#X_train = scaler.fit_transform(X_train)   # aprende + transforma
#X_val   = scaler.transform(X_val)         # solo transforma
#X_test  = scaler.transform(X_test)        # solo transforma
#
#print("\n=== Verificación del escalado (train) ===")
#print("Media por columna (≈0):", X_train.mean(axis=0).round(2))
#print("Std  por columna (≈1):", X_train.std(axis=0).round(2))
#


X_train = pd.read_csv('./xTrain.csv')

y_train = X_train['Label']
X_train = X_train.drop(['Label'], axis=1)
X_train = np.array(X_train)

X_val = pd.read_csv('./xVal.csv')

y_val = X_val['Label']
X_val = X_val.drop(['Label'], axis=1)
X_val = np.array(X_val)

X_test = pd.read_csv('./xTest.csv')

y_test = X_test['Label']
X_test = X_test.drop(['Label'], axis=1)
X_test = np.array(X_test)

# =============================================================================
# 5. TRAIN / VALIDATION — MLPClassifier
# =============================================================================
# MLP = Multi-Layer Perceptron. Es la red neuronal "clásica" de sklearn.
#
# Hiperparámetros clave:
#   hidden_layer_sizes : arquitectura de capas ocultas, p.ej. (64, 64)
#                        significa 2 capas con 64 neuronas cada una
#   max_iter           : número máximo de épocas (pasadas completas por el train)
#   n_iter_no_change   : early stopping — para si la loss no mejora en N épocas
#
# Triángulo de fitting:
#   Underfitting  → modelo demasiado simple, no aprende el patrón
#   Overfitting   → modelo demasiado complejo, memoriza el train pero falla en val
#   Generalización→ el punto óptimo entre ambos extremos

# --- Zona de experimentación (descomenta para probar distintos escenarios) ---

# Underfitting: arquitectura tiny + pocas épocas → accuracy bajo en train Y val
# epoch = 10
# architecture = (4, 4)

# Generalización: punto "dulce" para este dataset
epoch = 80
architecture = (90, 90)

# Overfitting: red enorme + muchas épocas → train muy alto, val cae
# epoch = 500
# architecture = (600, 600)

mlp = MLPClassifier(
    hidden_layer_sizes=architecture,
    max_iter=epoch,
    random_state=42,
    n_iter_no_change=50   # early stopping: para si no mejora en 50 épocas
)

print(f"\nEntrenando {epoch} épocas, arquitectura {architecture}...")
mlp.fit(X_train, y_train)

print('\n=== Arquitectura de la red ===')
print(f'  Capa de entrada  : ({mlp.n_features_in_} neuronas — una por feature)')
print(f'  Capas ocultas    : {architecture}')
print(f'  Capa de salida   : ({mlp.n_outputs_} neurona — HeartDisease 0/1)')

train_acc = accuracy_score(y_train, mlp.predict(X_train))
val_acc   = accuracy_score(y_val,   mlp.predict(X_val))

print(f'\n  Train accuracy : {train_acc:.2%}')
print(f'  Val   accuracy : {val_acc:.2%}')

# Diagnóstico rápido:
gap = train_acc - val_acc
if gap > 0.10:
    print("  → POSIBLE OVERFITTING: brecha train-val > 10 pp")
elif train_acc < 0.75:
    print("  → POSIBLE UNDERFITTING: accuracy de train muy bajo")
else:
    print("  → Modelo en zona de generalización razonable")


# =============================================================================
# 6. TEST — Evaluación final
# =============================================================================
# Solo se ejecuta UNA vez, después de haber fijado definitivamente los
# hiperparámetros. El resultado en test es el estimador más honesto del
# rendimiento real del modelo.

test_acc = accuracy_score(y_test, mlp.predict(X_test))
print(f'\n  Test  accuracy : {test_acc:.2%}')


# =============================================================================
# 7. VISUALIZACIÓN DE RESULTADOS
# =============================================================================
labels = ['Train', 'Validation', 'Test']
scores = [train_acc, val_acc, test_acc]
colors = ['#3498db', '#e67e22', '#2ecc71']

plt.figure(figsize=(8, 5))
bars = plt.bar(labels, scores, color=colors)
plt.ylim(0, 1.15)
plt.ylabel('Accuracy Score')
plt.title(f'Heart Failure Prediction — MLP {architecture}, {epoch} épocas')

for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2., height + 0.02,
             f'{height:.2%}', ha='center', fontweight='bold')

plt.tight_layout()
plt.show()