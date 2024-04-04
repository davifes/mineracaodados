import pandas as pd
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

    #target = 'situation'
    
    df = pd.read_csv(input_file,    # Nome do arquivo com dados
                     names=names, delimiter=';') # Nome das colunas                      

    # Converter as características para float
    df[features] = df[features].astype(float)
    
    # Converter a coluna "situation" para valores binários
    df['situation'] = df['situation'].map({'Failed': 0, 'Approved': 1})
    
    ShowInformationDataFrame(df, "Dataframe com 'situation' convertido para valores binários")
    
    # Plotar diagramas de dispersão para cada par de atributos
    plot_scatter_plots(df, features)
    
    # Calcular e plotar mapa de calor (heatmap) da correlação entre as colunas numéricas, incluindo 'situation'
    plot_correlation_heatmap(df)

def ShowInformationDataFrame(df, message=""):
    print(message+"\n")
    print(df.info())
    print(df.describe())
    print(df.head(10))
    print("\n") 

def plot_scatter_plots(df, features):
    for i in range(len(features)):
        for j in range(i + 1, len(features)):
            plt.figure(figsize=(8, 6))
            sns.scatterplot(data=df, x=features[i], y=features[j])
            plt.title(f'Diagrama de Dispersão entre {features[i]} e {features[j]}')
            plt.xlabel(features[i])
            plt.ylabel(features[j])
            plt.tight_layout()
            plt.show()

def plot_correlation_heatmap(df):
    corr = df.corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Mapa de Calor da Correlação')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
