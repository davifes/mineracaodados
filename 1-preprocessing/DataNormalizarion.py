import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

def main():
    # Faz a leitura do arquivo
    input_file = 'student-por.csv'
    output_file_zscore = 'output_file_zscore.csv'  # Nome do arquivo para Z-score normalization
    output_file_minmax = 'output_file_minmax.csv'  # Nome do arquivo para Min-Max normalization
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
    
    # Separating out the features
    x = df.loc[:, features].values
    
    # Separating out the target
    y = df.loc[:,[target]].values

    # Z-score normalization
    x_zcore = StandardScaler().fit_transform(x)
    normalized1Df = pd.DataFrame(data=x_zcore, columns=features)
    normalized1Df = pd.concat([normalized1Df, df[[target]]], axis=1)
    ShowInformationDataFrame(normalized1Df, "Dataframe Z-Score Normalized")
    

    # Mix-Max normalization
    x_minmax = MinMaxScaler().fit_transform(x)
    normalized2Df = pd.DataFrame(data=x_minmax, columns=features)
    normalized2Df = pd.concat([normalized2Df, df[[target]]], axis=1)
    ShowInformationDataFrame(normalized2Df, "Dataframe Min-Max Normalized")

    normalized1Df.to_csv(output_file_zscore, index=False)
    normalized2Df.to_csv(output_file_minmax, index=False)

def ShowInformationDataFrame(df, message=""):
    print(message+"\n")
    print(df.info())
    print(df.describe())
    print(df.head(10))
    print("\n") 

if __name__ == "__main__":
    main()