import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Carregar os dados
input_file = 'student-por.csv'

# Definir os nomes das colunas manualmente
names = ['school', 'sex', 'age', 'address', 'famsize', 'Pstatus', 'Medu', 'Fedu', 'Mjob', 'Fjob',
         'reason', 'guardian', 'traveltime', 'studytime', 'failures', 'schoolsup', 'famsup', 'paid', 
         'activities', 'nursery', 'higher', 'internet', 'romantic', 'famrel', 'freetime', 'goout', 
         'Dalc', 'Walc', 'health', 'absences', 'G1', 'G2', 'G3', 'situation']

# Ler o arquivo CSV com os nomes das colunas definidos
df = pd.read_csv(input_file, names=names, delimiter=';')

# Selecionar as variáveis numéricas para a distribuição de frequência
numeric_features = ['age', 'Medu', 'Fedu', 'traveltime', 'studytime', 'failures', 'famrel', 'freetime', 
                    'goout', 'Dalc', 'Walc', 'health', 'absences', 'G1', 'G2', 'G3']

# Inicializar listas para armazenar as coordenadas do gráfico 3D
x_coords = []
y_coords = []
z_coords = []

# Iterar sobre cada variável numérica para calcular as frequências
for feature in numeric_features:
    # Escolher o número de classes desejado
    num_classes = 10
    
    # Calcular a amplitude de classe (arredondado para cima)
    class_width = np.ceil((df[feature].max() - df[feature].min()) / num_classes)

    # Calcular o limite inferior da primeira classe
    lower_limit_first_class = df[feature].min()

    # Calcular os limites inferiores das classes subsequentes
    lower_limits = [lower_limit_first_class + i * class_width for i in range(num_classes)]

    # Calcular os limites superiores das classes
    upper_limits = [lower_limit + class_width for lower_limit in lower_limits]

    # Inicializar um dicionário para armazenar as frequências de cada classe
    frequency_distribution = {}

    # Percorrer o conjunto de dados para calcular as frequências
    for lower_limit, upper_limit in zip(lower_limits, upper_limits):
        frequency = ((df[feature] >= lower_limit) & (df[feature] < upper_limit)).sum()
        frequency_distribution[(lower_limit, upper_limit)] = frequency

    # Armazenar as coordenadas para o gráfico 3D
    for i, (limits, frequency) in enumerate(frequency_distribution.items()):
        x_coords.append((limits[0] + limits[1]) / 2)  # Média dos limites para x
        y_coords.append(frequency)  # Frequência para y
        z_coords.append(i)  # Classe para z

# Plotar gráfico de dispersão 3D
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x_coords, y_coords, z_coords)

ax.set_xlabel('Valores')
ax.set_ylabel('Frequência')
ax.set_zlabel('Classes')
ax.set_title('Gráfico de Dispersão 3D para Distribuição de Frequência de Variáveis Numéricas')
plt.show()
