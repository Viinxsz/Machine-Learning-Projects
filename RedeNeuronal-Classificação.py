# Importar as bibliotecas necessárias
import numpy as np
from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Carregar o dataset
lfw_people = fetch_lfw_people(min_faces_per_person=100, resize=0.4)
X, y = lfw_people.data, lfw_people.target
target_names = lfw_people.target_names
n_classes = target_names.shape[0]

# Analisar a dimensionalidade e características do dataset
print(f"Dimensões do dataset X: {X.shape}")
print(f"Dimensões do target y: {y.shape}")
print(f"Classes disponíveis: {target_names}")
print("\nPrimeiros exemplos do dataset:")
print(X[:5], "\n", y[:5])

# Dividir o dataset em subsets de treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalizar os dados
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Configuração da RNN inicial
rnn_initial = MLPClassifier(hidden_layer_sizes=(25, 25), activation='logistic', max_iter=1000, random_state=42)

# Treinamento da RNN
rnn_initial.fit(X_train, y_train)

# Avaliação da RNN inicial no conjunto de teste
accuracy_initial = rnn_initial.score(X_test, y_test)
print(f"Acurácia da RNN inicial no subset de teste: {accuracy_initial:.4f}")

# Definir os parâmetros para GridSearchCV
param_grid = {
    'hidden_layer_sizes': [(10,), (25,), (50,), (10, 10), (25, 25), (50, 50), (25, 10)],
    'alpha': [10**i for i in range(0, 8)]
}

# Configuração do modelo e do GridSearchCV
grid_search = GridSearchCV(
    MLPClassifier(activation='logistic', max_iter=1000, random_state=42),
    param_grid,
    cv=KFold(n_splits=5, shuffle=True, random_state=42),
    scoring='accuracy',
    n_jobs=-1
)

# Realizar o GridSearchCV
grid_search.fit(X_train, y_train)

# Melhor modelo encontrado
best_model = grid_search.best_estimator_
print(f"Melhor combinação de hiper-parâmetros: {grid_search.best_params_}")

# Avaliar a RNN otimizada no subset de teste
accuracy_best = best_model.score(X_test, y_test)
print(f"Acurácia da RNN otimizada no subset de teste: {accuracy_best:.4f}")

# Bônus: Comparar com um modelo de SVM
svm_model = SVC(kernel='linear', class_weight='balanced', random_state=42)
svm_model.fit(X_train, y_train)
accuracy_svm = svm_model.score(X_test, y_test)
print(f"Acurácia do modelo SVM no subset de teste: {accuracy_svm:.4f}")

# Representar alguns dos rostos previstas
n_samples_to_show = 5
fig, axes = plt.subplots(1, n_samples_to_show, figsize=(15, 8))
for i, ax in enumerate(axes):
    ax.imshow(X_test[i].reshape(lfw_people.images.shape[1], lfw_people.images.shape[2]), cmap='gray')
    predicted_label = best_model.predict([X_test[i]])[0]
    true_label = y_test[i]
    ax.set_title(f"Pred: {target_names[predicted_label]}\nTrue: {target_names[true_label]}")
    ax.axis("off")

plt.show()