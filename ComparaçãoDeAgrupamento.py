# Importando bibliotecas necessárias para o clustering
import numpy as np
import matplotlib.pyplot as plt
from sklearn import cluster, datasets, mixture
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from itertools import cycle, islice

# Preparando datasets de exemplo
np.random.seed(0)
n_samples = 1500

# Diferentes distribuições de dados para o clustering
datasets = [
    (datasets.make_blobs(n_samples=n_samples, random_state=8), {}),
    (datasets.make_moons(n_samples=n_samples, noise=0.05), {}),
    (datasets.make_circles(n_samples=n_samples, factor=0.5, noise=0.05), {}),
    (datasets.make_blobs(n_samples=n_samples, cluster_std=[1.0, 2.5, 0.5], random_state=170), {}),
    ((np.random.rand(n_samples, 2), None), {}),
    (datasets.make_blobs(n_samples=n_samples, random_state=170), {})
]

# Parâmetros para diferentes algoritmos de agrupamento
clustering_algorithms = [
    ("MiniBatchKMeans", cluster.MiniBatchKMeans(n_clusters=3)),
    ("AffinityPropagation", cluster.AffinityPropagation(damping=0.9)),
    ("MeanShift", cluster.MeanShift(bandwidth=2)),
    ("SpectralClustering",
     cluster.SpectralClustering(n_clusters=3, eigen_solver='arpack', affinity="nearest_neighbors")),
    ("Ward", cluster.AgglomerativeClustering(n_clusters=3, linkage='ward')),
    ("AgglomerativeClustering", cluster.AgglomerativeClustering(n_clusters=3)),
    ("DBSCAN", cluster.DBSCAN(eps=0.2)),
    ("OPTICS", cluster.OPTICS(min_samples=20, xi=0.05, min_cluster_size=0.1)),
    ("Birch", cluster.Birch(n_clusters=3)),
    ("GaussianMixture", mixture.GaussianMixture(n_components=3, covariance_type='full'))
]

# Executando e plotando resultados dos algoritmos nos diferentes datasets
plt.figure(figsize=(21, 9))
for i_dataset, (dataset, algo_params) in enumerate(datasets):
    X, y = dataset if len(dataset) == 2 else (dataset[0], None)
    X = StandardScaler().fit_transform(X)

    for name, algorithm in clustering_algorithms:
        algorithm.fit(X)
        if hasattr(algorithm, 'labels_'):
            y_pred = algorithm.labels_.astype(int)
        else:
            y_pred = algorithm.predict(X)

        plt.subplot(len(datasets), len(clustering_algorithms),
                    i_dataset * len(clustering_algorithms) + clustering_algorithms.index((name, algorithm)) + 1)
        plt.scatter(X[:, 0], X[:, 1], c=y_pred, s=10, cmap='viridis')
        plt.title(name, size=9)
plt.tight_layout()
plt.show()

"""Código para carregar e executar a demonstração dos pressupostos do K-means"""

# Configurando diferentes distribuições para observar os pressupostos do K-means
blobs_params = dict(random_state=42, n_samples=300)

# Dataset 1: Blobs com variância uniforme
X, _ = make_blobs(centers=4, cluster_std=1.0, **blobs_params)
plt.scatter(X[:, 0], X[:, 1], s=10)
plt.title("Dataset 1: Blobs Uniformes")
plt.show()

# Dataset 2: Blobs com variância heterogênea
X_varied, _ = make_blobs(centers=4, cluster_std=[0.5, 1.5, 1.0, 1.5], **blobs_params)
plt.scatter(X_varied[:, 0], X_varied[:, 1], s=10)
plt.title("Dataset 2: Blobs Variados")
plt.show()

# Dataset 3: Blobs com tamanhos desiguais
X_unbalanced, _ = make_blobs(cluster_std=[1.0, 2.5, 0.5], **blobs_params)
plt.scatter(X_unbalanced[:, 0], X_unbalanced[:, 1], s=10)
plt.title("Dataset 3: Blobs de Tamanhos Desiguais")
plt.show()

# Aplicando o K-means para observar os clusters formados
for X, title in [(X, "Blobs Uniformes"), (X_varied, "Blobs Variados"), (X_unbalanced, "Blobs de Tamanhos Desiguais")]:
    kmeans = KMeans(n_clusters=3)
    y_pred = kmeans.fit_predict(X)
    plt.scatter(X[:, 0], X[:, 1], c=y_pred, s=10, cmap='viridis')
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], c='red', marker='x')
    plt.title(f"K-means aplicado em: {title}")
    plt.show()
