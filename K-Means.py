# Importando as bibliotecas necessárias
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

# Criando o dataset sintético de agrupamento
n_samples = 1000
n_features = 2
centers = 5
cluster_std = [1.0, 1.5, 0.5, 2.0, 2.5]
random_state = 42

X, y_true = make_blobs(n_samples=n_samples, n_features=n_features, centers=centers, cluster_std=cluster_std,
                       random_state=random_state)

# Reordenando os dados aleatoriamente
indices = np.arange(n_samples)
np.random.shuffle(indices)
X = X[indices]
y_true = y_true[indices]

# Normalizando os dados
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Função para calcular a distância entre um ponto e um centroide
def dist_examples(x, centroide):
    """ Calcula a distância euclidiana entre o ponto x e o centroide no espaço n-dimensional """
    return np.linalg.norm(x - centroide)

# Definindo parâmetros para o K-means
n_clusters = 5
n_iter = 100
tol = 1e-3

# Inicializando os centroides com amostras aleatórias para garantir variabilidade
centroides = X[np.random.choice(n_samples, n_clusters, replace=False)]

for i in range(n_iter):
    # Atribuição de cada ponto ao centroide mais próximo
    polo_atribuido_exemplos = np.zeros(n_samples)
    for j in range(n_samples):
        distancias = [dist_examples(X[j], centroide) for centroide in centroides]
        polo_atribuido_exemplos[j] = np.argmin(distancias)

    # Atualizando cada centroide com a média dos pontos atribuídos a ele
    novo_centroides = np.zeros_like(centroides)
    for c in range(n_clusters):
        pontos_cluster = X[polo_atribuido_exemplos == c]
        if len(pontos_cluster) > 0:
            novo_centroides[c] = pontos_cluster.mean(axis=0)

    # Verificando convergência (se os centroides se moveram menos do que tol)
    if np.all(np.linalg.norm(novo_centroides - centroides, axis=1) < tol):
        print('Modelo converge na iteração:', i)
        break

    centroides = novo_centroides
else:
    print('Número máximo de iterações alcançado')

print('Centroides finais:')
print(centroides)

# Avaliando com o coeficiente de silhueta
if len(np.unique(polo_atribuido_exemplos)) > 1:  # Evita erro com um único cluster
    score_silhueta = silhouette_score(X, polo_atribuido_exemplos)
    print(f'Coeficiente de Silhueta: {score_silhueta}')
else:
    print("Erro: Apenas um único cluster encontrado; coeficiente de silhueta não é aplicável.")

# Representação gráfica do modelo formado
plt.figure(figsize=(10, 8))
plt.scatter(X[:, 0], X[:, 1], c=polo_atribuido_exemplos, cmap='viridis', s=50, alpha=0.6)
plt.scatter(centroides[:, 0], centroides[:, 1], c='red', marker='X', s=200, label='Centroides')
plt.legend()
plt.title("Modelo K-means com Agrupamento")
plt.show()

# Testando diferentes números de clusters para a regra da curva
n_clusters_range = range(2, 11)  # Começa em 2 para evitar o erro de único cluster
silhouette_scores = []

for n_c in n_clusters_range:
    melhor_score = -1
    for _ in range(5):  # Tentativas para inicialização aleatória
        # Inicializando centroides aleatoriamente
        centroides = X[np.random.choice(n_samples, n_c, replace=False)]

        # Implementação do K-means
        polo_atribuido_exemplos = np.zeros(n_samples)
        for i in range(n_iter):
            distancias = np.array(
                [[dist_examples(X[j], centroide) for centroide in centroides] for j in range(n_samples)])
            polo_atribuido_exemplos = np.argmin(distancias, axis=1)

            novo_centroides = np.array([X[polo_atribuido_exemplos == c].mean(axis=0) for c in range(n_c) if
                                        len(X[polo_atribuido_exemplos == c]) > 0])
            if novo_centroides.shape == centroides.shape and np.all(
                    np.linalg.norm(novo_centroides - centroides, axis=1) < tol):
                break
            centroides = novo_centroides

        # Avaliação pelo coeficiente de silhueta
        if len(np.unique(polo_atribuido_exemplos)) > 1:  # Evita erro com um único cluster
            score = silhouette_score(X, polo_atribuido_exemplos)
            if score > melhor_score:
                melhor_score = score

    silhouette_scores.append(melhor_score)

# Representando a regra da curva
plt.figure(figsize=(10, 6))
plt.plot(n_clusters_range, silhouette_scores, marker='o')
plt.xlabel("Número de Clusters")
plt.ylabel("Coeficiente de Silhueta")
plt.title("Regra da Curva para Seleção do Número de Clusters")
plt.show()
