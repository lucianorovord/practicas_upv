import matplotlib.pyplot as plt

#library scikit-learn
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


###############1. EDA:
#...


###############2. Preprocessing 
X, y = make_classification(n_samples=1000, n_features=20, n_informative=5, class_sep=0.8, random_state=42)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
#...


###############3. Train/Validation
###try and error with hyperparameters (tuning):
#underfitting
# epoch=10
# architecture = (4, 4)

#generalization
# epoch=25
# architecture = (128, 128)

#overfitting
#epoch = 100
# architecture = (600, 600)

epoch = 45
architecture = (199, 312)


###
mlp = MLPClassifier(hidden_layer_sizes=architecture, max_iter=epoch, random_state=42, n_iter_no_change=1000)
print(f"\nTraining for {epoch} epochs")
mlp.fit(X_train, y_train)
print('\nNeural network architecture')
print(f'input layer: ({mlp.n_features_in_})')
print(f'hidden layers: {architecture}')
print(f'output layer: ({mlp.n_outputs_})\n')
train_acc = accuracy_score(y_train, mlp.predict(X_train)) #(y, y')
val_acc = accuracy_score(y_val, mlp.predict(X_val)) #(y, y')


###############4. Test
test_acc = accuracy_score(y_test, mlp.predict(X_test)) #(y, y')


###############5. Results presentation
labels = ['Train', 'Validation', 'Test']
scores = [train_acc, val_acc, test_acc]
#figure
plt.figure(figsize=(8, 5))
bars = plt.bar(labels, scores, color=['#3498db', '#e67e22', '#2ecc71'])
plt.ylim(0, 1.1)
plt.ylabel('Accuracy Score')
plt.title('Final Model Accuracy Comparison')
for bar in bars: # Add text labels on top of bars
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.02,
             f'{height:.2%}', ha='center', fontweight='bold')
plt.show()
