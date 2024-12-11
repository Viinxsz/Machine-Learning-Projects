# Importar as bibliotecas necessárias
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

# Carregar o dataset
data = fetch_california_housing()
X, y = data.data, data.target

# Analisar a dimensionalidade e características do dataset
print(f"Dimensões do dataset X: {X.shape}")
print(f"Dimensões do target y: {y.shape}")
print(f"Descrição das features:\n{data.DESCR}")
print("\nPrimeiros exemplos do dataset:")
print(X[:5], "\n", y[:5])

# Reordenar aleatoriamente e dividir os dados
X, y = X[np.random.permutation(len(y))], y

# Dividir o dataset em subsets de treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalizar os dados
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Configuração da RNN inicial
rnn_initial = MLPRegressor(hidden_layer_sizes=(25, 25), activation='relu', max_iter=1000, random_state=42)

# Treinamento da RNN
rnn_initial.fit(X_train, y_train)

# Avaliação da RNN inicial no conjunto de teste
r2_initial = rnn_initial.score(X_test, y_test)
print(f"R² da RNN inicial no subset de teste: {r2_initial:.4f}")

# Definir os parâmetros para GridSearchCV
param_grid = {
    'hidden_layer_sizes': [(10,), (25,), (50,), (10, 10), (25, 25), (50, 50), (25, 10)],
    'alpha': [10**i for i in range(0, 8)]
}

# Configuração do modelo e do GridSearchCV
grid_search = GridSearchCV(
    MLPRegressor(activation='relu', max_iter=1000, random_state=42),
    param_grid,
    cv=KFold(n_splits=5, shuffle=True, random_state=42),
    scoring='r2',
    n_jobs=-1
)

# Realizar o GridSearchCV
grid_search.fit(X_train, y_train)

# Melhor modelo encontrado
best_model = grid_search.best_estimator_
print(f"Melhor combinação de hiper-parâmetros: {grid_search.best_params_}")

# Avaliar a RNN otimizada no subset de teste
r2_best = best_model.score(X_test, y_test)
print(f"R² da RNN otimizada no subset de teste: {r2_best:.4f}")
