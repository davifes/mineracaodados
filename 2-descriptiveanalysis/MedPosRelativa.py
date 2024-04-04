import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def main():
    # Faz a leitura do arquivo
    input_file = 'student-por.csv'
    names = ['school', 'sex', 'age', 'address', 'famsize', 'Pstatus', 'Medu', 'Fedu', 'Mjob', 'Fjob',
    'reason', 'guardian', 'traveltime', 'studytime', 'failures', 'schoolsup', 'famsup', 'paid', 'activities', 
    'nursery', 'higher', 'internet', 'romantic', 'famrel', 'freetime', 'goout', 'Dalc', 'Walc', 'health',
    'absences', 'G1', 'G2', 'G3', 'situation']
    features = ['age', 'Medu', 'Fedu', 'traveltime', 'studytime', 'failures', 'famrel', 'freetime', 'goout', 'Dalc', 'Walc', 'health',
    'absences', 'G1', 'G2', 'G3'] 

    target = 'situation'
    
    df = pd.read_csv(input_file,    # Nome do arquivo com dados
                     names=names, delimiter=';') # Nome das colunas                      
    ShowInformationDataFrame(df, "Dataframe original")

    # Converter as características para float
    df[features] = df[features].astype(float)
    
    # Calculate dispersion measures
    calculate_dispersion_measures(df, features)
    
    # Calculate Z-score
    calculate_z_score(df, features)
    
    # Calculate quartiles
    calculate_quartiles(df, features)
    
    # Plotar boxplots separados para cada característica
    plot_boxplots(df, features)

def ShowInformationDataFrame(df, message=""):
    print(message+"\n")
    print(df.info())
    print(df.describe())
    print(df.head(10))
    print("\n") 

def calculate_dispersion_measures(df, features):
    for feature in features:
        amplitude = df[feature].max() - df[feature].min()
        std_dev = df[feature].std()
        print(f"Feature: {feature}")
        print(f"Amplitude: {amplitude}")
        print(f"Desvio Padrao: {std_dev}\n")

def calculate_z_score(df, features):
    for feature in features:
        mean = df[feature].mean()
        std_dev = df[feature].std()
        df[f'{feature}_z_score'] = (df[feature] - mean) / std_dev
        print(f"Feature: {feature}")
        print(df[f'{feature}_z_score'])

def calculate_quartiles(df, features):
    for feature in features:
        print(f"Feature: {feature}")
        print(df[feature].quantile([0.25, 0.5, 0.75]))

def plot_boxplots(df, features):
    for feature in features:
        df.boxplot(column=feature)
        plt.title(f'Boxplot de {feature}')
        plt.ylabel('Valores')
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    main()
