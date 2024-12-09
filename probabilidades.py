import numpy as np
import matplotlib.pyplot as plt


# Função para calcular a probabilidade pelo Teorema de Bayes
def bayes_theorem(p_a, p_b_dado_a, p_b_dado_no_a):
    """
    Devolver a probabilidade de A dado B segundo o Teorema de Bayes

    p_a -- probabilidade a priori de A
    p_b_dado_a -- probabilidade de B dado A
    p_b_dado_no_a -- probabilidade de B dado não A

    Devolver:
    p_a_dado_b -- probabilidade de A dado B
    """
    # P(B)
    p_b = (p_b_dado_a * p_a) + (p_b_dado_no_a * (1 - p_a))

    # P(A|B)
    p_a_dado_b = (p_b_dado_a * p_a) / p_b

    return p_a_dado_b
# Resolver os exemplos fornecidos
print("Exemplos resolvidos:")

# Exemplo 1
p_a = 0.01
p_b_dado_a = 0.9
p_b_dado_no_a = 0.2
p_a_dado_b = bayes_theorem(p_a, p_b_dado_a, p_b_dado_no_a)
print("Exemplo 1:")
print('Probabilidade de A dado B:', round(p_a_dado_b, 4))  # 0.0435

# Exemplo 2
p_a = 0.01
p_b_dado_a = 0.85
p_b_dado_no_a = 0.15
p_a_dado_b = bayes_theorem(p_a, p_b_dado_a, p_b_dado_no_a)
print("\nExemplo 2:")
print('Probabilidade de A dado B:', round(p_a_dado_b, 4))  # 0.0541

# Exemplo 3
p_a = 0.01
p_b_dado_a = 0.9
p_b_dado_no_a = 0.05
p_a_dado_b = bayes_theorem(p_a, p_b_dado_a, p_b_dado_no_a)
print("\nExemplo 3:")
print('Probabilidade de A dado B:', round(p_a_dado_b, 4))  # 0.1538
# Probabilidade a priori
p_a = 0.01

# Array de sensibilidade e especificidade
sensibilidade = np.linspace(0.65, 0.95, 10)  # +/- 25% sobre o valor base de 0.9
especificidade = np.linspace(0.05, 0.35, 10)  # +/- 25% sobre o valor base de 0.2

# Cálculo da probabilidade a posteriori com sensibilidade variável e especificidade fixa
posteriori_sens = [bayes_theorem(p_a, s, 0.2) for s in sensibilidade]
posteriori_esp = [bayes_theorem(p_a, 0.9, e) for e in especificidade]
# Plotar os resultados
fig, axs = plt.subplots(2, 1, figsize=(10, 8))
# Plot Sensibilidade
axs[0].plot(sensibilidade, posteriori_sens, marker='o', color='b')
axs[0].set_title("Variação da probabilidade a posteriori com Sensibilidade")
axs[0].set_xlabel("Sensibilidade")
axs[0].set_ylabel("Probabilidade a posteriori")

# Plot Especificidade
axs[1].plot(especificidade, posteriori_esp, marker='o', color='r')
axs[1].set_title("Variação da probabilidade a posteriori com Especificidade")
axs[1].set_xlabel("Especificidade")
axs[1].set_ylabel("Probabilidade a posteriori")

plt.tight_layout()
plt.show()
