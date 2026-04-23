import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score



# Hacer la separacion de los datos en 3 CSV para TRAINING, VALIDATION y TESTING
# Únicamente en este archivo empezar En los apartados 3)Training y 4)Testing a partir de estos 3 CSV: (guía: hfp_pipeline.py)
#
# xTrain_cs, xVal_cs, xTest_cs

X_train = pd.read_csv('./CyberSecurity/cs_train.csv')
y_train = X_train['Label']
X_train = X_train.drop(['Label'], axis=1)
X_train = np.array(X_train)

X_val = pd.read_csv('./CyberSecurity/cs_val.csv')
y_val = X_val['Label']
X_val = X_val.drop(['Label'], axis=1)
X_val = np.array(X_val)

X_test = pd.read_csv('./CyberSecurity/cs_test.csv')

y_test = X_test['Label']
X_test = X_test.drop(['Label'], axis=1)
X_test = np.array(X_test)

# =============================================================================
# 5. TRAIN / VALIDATION — MLPClassifier
# =============================================================================

epoch = 60
architecture = (80, 50)

mlp = MLPClassifier(
    hidden_layer_sizes=architecture,
    max_iter=epoch,
    random_state=42,
    n_iter_no_change=50,   # early stopping: para si no mejora en 50 épocas
    verbose=True
)

print(f"\nEntrenando {epoch} épocas, arquitectura {architecture}...")
mlp.fit(X_train, y_train)

print('\n=== Arquitectura de la red ===')
print(f'  Capa de entrada  : ({mlp.n_features_in_} neuronas — una por feature)')
print(f'  Capas ocultas    : {architecture}')
print(f'  Capa de salida   : ({mlp.n_outputs_} neurona — CyberSecurity 0/1)')

# Predictions
y_train_pred = mlp.predict(X_train)
y_val_pred = mlp.predict(X_val)
y_test_pred = mlp.predict(X_test)

# Accuracy
train_acc = accuracy_score(y_train, mlp.predict(X_train))
val_acc   = accuracy_score(y_val,   mlp.predict(X_val))

# Precision
train_prec = precision_score(y_train, y_train_pred)
val_prec   = precision_score(y_val,   y_val_pred)
test_prec  = precision_score(y_test,  y_test_pred)

# Recall
train_rec = recall_score(y_train, y_train_pred)
val_rec   = recall_score(y_val,   y_val_pred)
test_rec  = recall_score(y_test,  y_test_pred)


# Confusion Matrix 
cm_train = confusion_matrix(y_train, y_train_pred)
cm_val = confusion_matrix(y_val, y_val_pred)
cm_test = confusion_matrix(y_test, y_test_pred)

############### Mostrar matrices (forma recomendada)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

ConfusionMatrixDisplay(cm_train).plot(ax=axes[0])
axes[0].set_title("Train")

ConfusionMatrixDisplay(cm_val).plot(ax=axes[1])
axes[1].set_title("Validacion")

ConfusionMatrixDisplay(cm_test).plot(ax=axes[2])
axes[2].set_title("Test")

plt.tight_layout()
plt.show()



############### TP, FP, TN, FN (ejemplo con TEST)

tn, fp, fn, tp = cm_test.ravel()

labels = ['TP', 'FP', 'TN', 'FN']
values = [tp, fp, tn, fn]

plt.figure(figsize=(6,4))
bars = plt.bar(labels, values)

for bar in bars: 
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{int(height)}', ha= 'center', va='bottom')
    
plt.title('Confusion Matrix (TEST)')
plt.ylabel('Count')
plt.show()


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

test_acc = accuracy_score(y_test, mlp.predict(X_test))
print(f'\n  Test  accuracy : {test_acc:.2%}')

# =============================================================================
# 7. VISUALIZACIÓN DE RESULTADOS
# =============================================================================

labels_acc = ['Train', 'Validation', 'Test']
scores_acc = [train_acc, val_acc, test_acc]
precision = [train_prec, val_prec, test_prec]
recall = [train_rec, val_rec, test_rec]



x = np.arange(len(labels_acc))  
ancho = 0.25


plt.figure(figsize=(10, 6))

barras_acc  = plt.bar(x - ancho, scores_acc,  ancho, label='Accuracy',color='#3498db')
barras_prec = plt.bar(x,         precision, ancho, label='Precision',color='#e67e22')
barras_rec  = plt.bar(x + ancho, recall,    ancho, label='Recall',color='#2ecc71')

plt.ylim(0, 1.15)
plt.ylabel('Accuracy Score')
plt.title(f'CyberSecurity — MLP {architecture}, {epoch} épocas')

for bar in barras_acc:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
             f'{height:.2%}', ha='center', fontsize=8, fontweight='bold')

for bar in barras_prec:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
             f'{height:.2%}', ha='center', fontsize=8, fontweight='bold')

for bar in barras_rec:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
             f'{height:.2%}', ha='center', fontsize=8, fontweight='bold')
    
plt.xticks(x, labels_acc)
plt.legend()
plt.tight_layout()
plt.show()