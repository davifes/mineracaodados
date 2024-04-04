"import pandas as pd
import seaborn as sns
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

    # Selecionar apenas as características numéricas
    df_numeric = df[features]
    
    # Converter as características para float
    df_numeric = df_numeric.astype(float)
    
    # Gerar matriz de correlação
    correlation_matrix = df_numeric.corr()
    print("Matriz de Correlação:")
    print(correlation_matrix)

    # Plotar mapa de calor da matriz de correlação
    plot_correlation_heatmap(correlation_matrix)

def ShowInformationDataFrame(df, message=""):
    print(message+"\n")
    print(df.info())
    print(df.describe())
    print(df.head(10))
    print("\n") 

def plot_correlation_heatmap(correlation_matrix):
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Matriz de Correlação')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
