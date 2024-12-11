# Importando as bibliotecas necessárias
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.metrics import silhouette_score, accuracy_score
from mpl_toolkits.mplot3d import Axes3D

# Carregando o dataset Iris
iris = load_iris()
X = iris.data
y = iris.target

# Representação em 3D das primeiras três características
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y, cmap='viridis', s=50)
ax.set_xlabel('Comprimento das sépalas')
ax.set_ylabel('Largura das sépalas')
ax.set_zlabel('Comprimento das pétalas')
plt.title('Representação 3D das primeiras três características do dataset Iris')
plt.show()

# Pré-processamento dos dados
# Reordenando e normalizando
indices = np.arange(len(X))
np.random.shuffle(indices)
X = X[indices]
y = y[indices]

scaler = StandardScaler()
X = scaler.fit_transform(X)

# Dividindo em conjuntos de treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Formando o modelo de agrupamento K-means com 3 clusters
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X_train)

# Avaliação do modelo de K-means no conjunto de teste
y_kmeans = kmeans.predict(X_test)
silhouette_avg = silhouette_score(X_test, y_kmeans)
print(f'Coeficiente de Silhueta do modelo K-means: {silhouette_avg}')

# Representação gráfica dos resultados do K-means
plt.figure(figsize=(8, 6))
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_kmeans, cmap='viridis', s=50, alpha=0.7, label='Clusters')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=200, c='red', label='Centroides')
plt.xlabel('Característica 1')
plt.ylabel('Característica 2')
plt.legend()
plt.title('Resultados do K-means no conjunto de teste')
plt.show()

# Modelo de classificação SVM com GridSearchCV para otimização de hiperparâmetros
param_grid = {
    'C': [0.1, 1, 10, 100],
    'kernel': ['linear', 'rbf', 'poly']
}
svc = SVC(random_state=42)
grid_search = GridSearchCV(svc, param_grid, cv=5)
grid_search.fit(X_train, y_train)

# Avaliação do melhor modelo SVM no conjunto de teste
best_svc = grid_search.best_estimator_
y_svc = best_svc.predict(X_test)
accuracy = accuracy_score(y_test, y_svc)
print(f'Acurácia do modelo SVM: {accuracy:.2f}')
print(f'Melhores parâmetros SVM: {grid_search.best_params_}')

# Representação gráfica das previsões do SVM
plt.figure(figsize=(8, 6))
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_svc, cmap='viridis', s=50, alpha=0.7)
plt.xlabel('Característica 1')
plt.ylabel('Característica 2')
plt.title('Previsões do SVM no conjunto de teste')
plt.show()

# Comparação dos resultados
# Avaliação da correspondência entre clusters K-means e classes SVM
print("Comparação entre agrupamento (K-means) e classificação (SVM):")
print(" - Coeficiente de Silhueta (K-means):", silhouette_avg)
print(" - Acurácia (SVM):", accuracy)

# Conclusão sobre a divisão do espaço e correspondência dos clusters vs. classes
