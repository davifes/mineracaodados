import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def main():
    # Definindo os nomes das colunas
    names = ['school', 'sex', 'age', 'address', 'famsize', 'Pstatus', 'Medu', 'Fedu', 'Mjob', 'Fjob',
             'reason', 'guardian', 'traveltime', 'studytime', 'failures', 'schoolsup', 'famsup', 'paid',
             'activities', 'nursery', 'higher', 'internet', 'romantic', 'famrel', 'freetime', 'goout',
             'Dalc', 'Walc', 'health', 'absences', 'G1', 'G2', 'G3', 'situation']
    
    # Definindo as features
    features = ['age', 'Medu', 'Fedu', 'traveltime', 'studytime', 'failures', 'famrel', 'freetime', 'goout',
                'Dalc', 'Walc', 'health', 'absences', 'G1', 'G2', 'G3', 'school', 'sex', 'address', 'famsize',
                'Pstatus', 'Mjob', 'Fjob', 'reason', 'guardian', 'schoolsup', 'famsup', 'paid', 'activities',
                'nursery', 'higher', 'internet', 'romantic', 'situation']
    
    # Arquivo de saída
    output_file = 'studentClear.csv'
    
    # Arquivo de entrada
    input_file = 'student-por.csv'
    
    # Leitura do arquivo CSV
    df = pd.read_csv(input_file, names=names, usecols=features, na_values='?', delimiter=';')
    
    # Criando uma cópia do DataFrame original
    df_original = df.copy()
    
    # Imprimindo as 15 primeiras linhas do arquivo
    print("PRIMEIRAS 15 LINHAS\n")
    print(df.head(15))
    print("\n")
    
    # Imprimindo informações gerais sobre os dados
    print("INFORMAÇÕES GERAIS DOS DADOS\n")
    print(df.info())
    print("\n")
    
    # Imprimindo uma análise descritiva dos dados
    print("DESCRIÇÃO DOS DADOS\n")
    print(df.describe())
    print("\n")
    
    # Imprimindo a quantidade de valores faltantes por coluna
    print("VALORES FALTANTES\n")
    print(df.isnull().sum())
    print("\n")
    
    # Atualizando os valores faltantes
    columns_missing_value = df.columns[df.isnull().any()]
    method = 'mode'  # number or median or mean or mode
    
    for c in columns_missing_value:
        UpdateMissingValues(df, c)
    
    # Imprimindo a descrição após a atualização
    print(df.describe())
    print("\n")
    
    # Imprimindo as primeiras 15 linhas após a atualização
    print(df.head(15))
    print(df_original.head(15))
    print("\n")
    
    # Salvando o arquivo com o tratamento para dados faltantes
    df.to_csv(output_file, header=False, index=False)
    
    # Plotando as estatísticas
    plot_statistics(df)


def UpdateMissingValues(df, column, method="mode", number=0):
    if method == 'number':
        # Substituindo valores ausentes por um número
        df[column].fillna(number, inplace=True)
    elif method == 'median':
        # Substituindo valores ausentes pela mediana
        median = df[column].median()
        df[column].fillna(median, inplace=True)
    elif method == 'mean':
        # Substituindo valores ausentes pela média
        mean = df[column].mean()
        df[column].fillna(mean, inplace=True)
    elif method == 'mode':
        # Substituindo valores ausentes pela moda
        mode = df[column].mode()[0]
        df[column].fillna(mode, inplace=True)


def plot_statistics(df):
    numeric_features = ['age', 'Medu', 'Fedu', 'traveltime', 'studytime', 'failures', 'famrel', 'freetime',
                        'goout', 'Dalc', 'Walc', 'health', 'absences', 'G1', 'G2', 'G3']
    nominal_features = ['school', 'sex', 'address', 'famsize', 'Pstatus', 'Mjob', 'Fjob', 'reason', 'guardian',
                        'schoolsup', 'famsup', 'paid', 'activities', 'nursery', 'higher', 'internet', 'romantic',
                        'situation']

    for feature in numeric_features:
        plt.figure(figsize=(10, 5))
        plt.title(f'Distribution of {feature}')
        plt.hist(df[feature], bins=20, alpha=0.5, color='blue', edgecolor='black', linewidth=1.5)

        mean_val = df[feature].mean()
        median_val = df[feature].median()
        mode_val = df[feature].mode().values[0]

        plt.axvline(mean_val, color='red', linestyle='dashed', linewidth=2, label=f'Mean: {mean_val:.2f}')
        plt.axvline(median_val, color='green', linestyle='dashed', linewidth=2, label=f'Median: {median_val:.2f}')
        plt.axvline(mode_val, color='purple', linestyle='dashed', linewidth=2, label=f'Mode: {mode_val:.2f}')

        plt.legend()
        plt.xlabel(feature)
        plt.ylabel('Frequency')
        plt.show()

    for feature in nominal_features:
        plt.figure(figsize=(8, 4))
        plt.title(f'Mode of {feature}')
        plt.bar(df[feature].value_counts().index, df[feature].value_counts().values, color='skyblue')
        plt.xlabel(feature)
        plt.ylabel('Frequency')
        plt.xticks(rotation=45)
        plt.show()

if __name__ == "__main__":
    main()
