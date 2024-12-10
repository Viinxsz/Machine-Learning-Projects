# Regressão logística: Regularização e previsões
# Importar bibliotecas necessárias
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification, load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import classification_report, confusion_matrix

# Criar um dataset sintético para regressão logística
# TODO: Gerar um dataset sintético, com o termo de bias e erro de forma manual
m = 1000  # Número de amostras
n = 2     # Número de características

# Gerar um array 2D m x n com valores números aleatórios entre -1 e 1
X = np.random.rand(m, n) * 2 - 1  # Valores entre -1 e 1

# Inserir o termo de bias como primeira coluna de 1s
X = np.hstack((np.ones((m, 1)), X))  # Adiciona uma coluna de 1's

# Gerar um array de theta de n + 1 valores aleatórios
Theta_verd = np.random.rand(n + 1)  # Theta verdadeiro

# Calcular Y em função de X e Theta_verd
# Adicionar um termo de erro modificável
error = 0.20
Y = X @ Theta_verd + np.random.randn(m) * error  # Termo de erro
Y = 1 / (1 + np.exp(-Y))  # Transformar com a função sigmoide
Y = (Y >= 0.5).astype(float)  # Transformar Y para valores de 0 e 1

# Verificar os valores e dimensões dos vetores
print('Theta a estimar:')
print(Theta_verd)

print('Primeiras 10 filas e 5 colunas de X e Y:')
print(X[:10])
print(Y[:10])

print('Dimensões de X e Y:')
print(X.shape)
print(Y.shape)

# Implementar a função sigmoide
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Pré-processar os dados
# Reordenar o dataset aleatoriamente
print('Primeiras 10 filas e 5 colunas de X e Y antes da reorganização:')
print(X[:10])
print(Y[:10])

# Reorganizar os dados no dataset X e Y
np.random.seed(42)  # Para reprodutibilidade
indices = np.random.permutation(m)
X, Y = X[indices], Y[indices]

print('Primeiras 10 filas e 5 colunas de X e Y após a reorganização:')
print(X[:10])
print(Y[:10])

# Normalizar o dataset
def normalize(x, mu, std):
    """ Normalizar um dataset com exemplos X """
    return (x - mu) / std

# Encontrar a média e o desvio padrão das características de X
mu = np.mean(X[:, 1:], axis=0)  # Excluir a coluna de bias
std = np.std(X[:, 1:], axis=0)

print('Média e desvio típico das características:')
print(mu)
print(std)

# Normalizar o dataset
X_norm = np.copy(X)
X_norm[:, 1:] = normalize(X[:, 1:], mu, std)  # Normalizar apenas as colunas 1 e seguintes

print('X normalizada:')
print(X_norm[:10])

# Dividir os dataset em subset de formação, CV e testes
ratios = [60, 20, 20]
print('Ratios:\n', ratios)

r = [0, 0]
r[0] = round(m * ratios[0] / 100)
r[1] = r[0] + round(m * ratios[1] / 100)
print('Índices de corte:\n', r)

X_train, X_cv, X_test = np.split(X_norm, [r[0], r[1]])
Y_train, Y_cv, Y_test = np.split(Y, [r[0], r[1]])

print('Tamanhos dos subsets:')
print(X_train.shape, Y_train.shape)
print(X_cv.shape, Y_cv.shape)
print(X_test.shape, Y_test.shape)

# Formar um modelo inicial sobre o subset de formação
# Função de custo e gradient descent
def cost_function(theta, X, Y, lambda_):
    m = len(Y)
    h = sigmoid(X @ theta)
    return (-1/m) * (Y @ np.log(h) + (1 - Y) @ np.log(1 - h)) + (lambda_ / (2 * m)) * np.sum(theta[1:] ** 2)

def gradient(theta, X, Y, lambda_):
    m = len(Y)
    h = sigmoid(X @ theta)
    grad = (1/m) * (X.T @ (h - Y)) + (lambda_ / m) * np.r_([0, theta[1:]])  # Regularização
    return grad

# Gradient descent
def gradient_descent(X, Y, alpha, lambda_, iterations):
    m, n = X.shape
    theta = np.zeros(n)
    j_history = np.zeros(iterations)

    for i in range(iterations):
        theta -= alpha * gradient(theta, X, Y, lambda_)
        j_history[i] = cost_function(theta, X, Y, lambda_)

    return theta, j_history

# Parâmetros
alpha = 0.1
lambda_ = 0.1
iterations = 1000

# Treinando o modelo
theta_initial = np.zeros(X_train.shape[1])
theta_train, j_history_train = gradient_descent(X_train, Y_train, alpha, lambda_, iterations)

# Representar a evolução da função de custo
plt.figure(1)
plt.plot(range(iterations), j_history_train)
plt.title('Evolução da função de custo (Training Set)')
plt.xlabel('Número de Iterações')
plt.ylabel('Custo')
plt.grid(True)
plt.show()

# Comprovar a adequação do modelo
Y_train_pred = sigmoid(X_train @ theta_train) >= 0.5
Y_cv_pred = sigmoid(X_cv @ theta_train) >= 0.5

print('Classificação no conjunto de treino:')
print(classification_report(Y_train, Y_train_pred))
print('Classificação no conjunto de validação:')
print(classification_report(Y_cv, Y_cv_pred))

# Encontrar o hiper-parâmetro lambda ótimo por CV
lambdas = [0., 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1e0, 3e0, 1e1]
j_cv = []

for lambda_ in lambdas:
    theta_cv, _ = gradient_descent(X_train, Y_train, alpha, lambda_, iterations)
    cost_cv = cost_function(theta_cv, X_cv, Y_cv, lambda_)
    j_cv.append(cost_cv)

# Representar graficamente o erro final para cada valor de lambda
plt.figure(2)
plt.plot(lambdas, j_cv, marker='o')
plt.xscale('log')
plt.title('Erro de validação cruzada para diferentes valores de lambda')
plt.xlabel('Lambda')
plt.ylabel('Custo')
plt.grid(True)
plt.show()

# Escolher o modelo ótimo e o valor lambda, com o menor erro no subset do CV
lambda_optimal = lambdas[np.argmin(j_cv)]
theta_final, _ = gradient_descent(X_train, Y_train, alpha, lambda_optimal)
print('Valor ótimo de lambda:', lambda_optimal)

# Avaliar o modelo sobre o subset de teste
j_test = cost_function(theta_final, X_test, Y_test, lambda_optimal)
print('Custo no subset de teste:', j_test)

Y_test_pred = sigmoid(X_test @ theta_final) >= 0.5
print('Classificação no conjunto de teste:')
print(classification_report(Y_test, Y_test_pred))

# Fazer previsões sobre novos exemplos
X_new = np.array([[1, 0.5, -0.2]])  # Novo exemplo com bias
X_new[:, 1:] = normalize(X_new[:, 1:], mu, std)  # Normalizar

Y_new_pred = sigmoid(X_new @ theta_final) >= 0.5
print('Previsão para o novo exemplo:', Y_new_pred)

