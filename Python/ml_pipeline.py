import matplotlib.pyplot as plt

# library scikit-learn
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay


############### 1. EDA:
# (omitido)


############### 2. Preprocessing 
X, y = make_classification(
    n_samples=1000,
    n_features=20,
    n_informative=5,
    class_sep=0.8,
    random_state=42
)

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)


############### 3. Train/Validation

epoch = 2000
architecture = (55, 55)

mlp = MLPClassifier(
    hidden_layer_sizes=architecture,
    max_iter=epoch,
    random_state=42,
    n_iter_no_change=1000
)

print(f"\nTraining for {epoch} epochs")
mlp.fit(X_train, y_train)

print('\nNeural network architecture')
print(f'input layer: ({mlp.n_features_in_})')
print(f'hidden layers: {architecture}')
print(f'output layer: ({mlp.n_outputs_})\n')


# Predictions
y_train_pred = mlp.predict(X_train)
y_val_pred = mlp.predict(X_val)
y_test_pred = mlp.predict(X_test)

# Accuracy
train_acc = accuracy_score(y_train, y_train_pred)
val_acc = accuracy_score(y_val, y_val_pred)
test_acc = accuracy_score(y_test, y_test_pred)


############### 4. Confusion Matrix

cm_train = confusion_matrix(y_train, y_train_pred)
cm_val = confusion_matrix(y_val, y_val_pred)
cm_test = confusion_matrix(y_test, y_test_pred)


############### 5. Mostrar matrices (forma recomendada)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

ConfusionMatrixDisplay(cm_train).plot(ax=axes[0])
axes[0].set_title("Train")

ConfusionMatrixDisplay(cm_val).plot(ax=axes[1])
axes[1].set_title("Validation")

ConfusionMatrixDisplay(cm_test).plot(ax=axes[2])
axes[2].set_title("Test")

plt.tight_layout()
plt.show()


############### 6. Extra: TP, FP, TN, FN (ejemplo con TEST)

tn, fp, fn, tp = cm_test.ravel()

labels = ['TP', 'FP', 'TN', 'FN']
values = [tp, fp, tn, fn]

plt.figure(figsize=(6, 4))
bars = plt.bar(labels, values)

for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{int(height)}', ha='center', va='bottom')

plt.title('Confusion Matrix Breakdown (Test)')
plt.ylabel('Count')
plt.show()


############### 7. Accuracy comparison (esto sí estaba bien planteado)

labels_acc = ['Train', 'Validation', 'Test']
scores_acc = [train_acc, val_acc, test_acc]

plt.figure(figsize=(6, 4))
bars = plt.bar(labels_acc, scores_acc)

plt.ylim(0, 1.1)
plt.ylabel('Accuracy')
plt.title('Model Accuracy Comparison')

for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.2%}', ha='center')

plt.show()