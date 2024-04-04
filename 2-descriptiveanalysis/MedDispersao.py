import pandas as pd
import numpy as np

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

if __name__ == "__main__":
    main()