import pandas as pd
from math import sqrt

# Função de cálculo da distância euclidiana
def distancia_euclideana(vet1, vet2):
    distancia = 0
    for i in range(len(vet1)):
        distancia += (vet1[i] - vet2[i])**2
    return sqrt(distancia)

# Função que retorna os k vizinhos mais próximos
def retorna_vizinhos(base_treinamento, amostra_teste, num_vizinhos):
    distancias = list()
    for linha_tre in base_treinamento:
        dist = distancia_euclideana(amostra_teste, linha_tre[:-1])
        distancias.append((linha_tre, dist))
    distancias.sort(key=lambda tup: tup[1])
    vizinhos = list()
    for i in range(num_vizinhos):
        vizinhos.append(distancias[i][0])
    return vizinhos

# Função de predição/classificação
def classifica(base_treinamento, amostra_teste, num_vizinhos):
    vizinhos = retorna_vizinhos(base_treinamento, amostra_teste, num_vizinhos)
    rotulos = [v[-1] for v in vizinhos]
    predicao = max(set(rotulos), key=rotulos.count)
    return predicao

# Carregando e processando a base de dados
input_file = 'student-por.csv'
names = ['school', 'sex', 'age', 'address', 'famsize', 'Pstatus', 'Medu', 'Fedu', 'Mjob', 'Fjob',
         'reason', 'guardian', 'traveltime', 'studytime', 'failures', 'schoolsup', 'famsup', 'paid', 
         'activities', 'nursery', 'higher', 'internet', 'romantic', 'famrel', 'freetime', 'goout', 
         'Dalc', 'Walc', 'health', 'absences', 'G1', 'G2', 'G3', 'situation']

features = ['age', 'Medu', 'Fedu', 'traveltime', 'studytime', 'failures', 'famrel', 'freetime', 
            'goout', 'Dalc', 'Walc', 'health', 'absences', 'G1', 'G2', 'G3'] 

target = 'situation'

# Leitura dos dados
data = pd.read_csv(input_file, names=names, delimiter=';')

# Separando features e target
data_features = data[features].values.tolist()
data_target = data[target].values.tolist()

# Combinando features e target para formar a base de treinamento
dataset = [data_features[i] + [data_target[i]] for i in range(len(data_features))]

# Definindo uma amostra de teste (exemplo)
amostra = [18, 3, 2, 1, 2, 0, 4, 3, 4, 1, 1, 5, 10, 10, 11, 12]

# Classificação usando o KNN
num_vizinhos = 3
predicao = classifica(dataset, amostra, num_vizinhos)

print('Resultado da classificação')
print('Predição:', predicao)
