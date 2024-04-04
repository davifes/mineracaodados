import pandas as pd

def complete_missing_data(input_file_path, output_file_path):
    # Leitura do arquivo CSV
    df = pd.read_csv(input_file_path)

    # Preenchimento dos dados faltantes
    for column in df.columns:
        if df[column].dtype == 'object':  # Se for uma coluna de texto
            mode_value = df[column].mode()[0]  # Calcula a moda
            df[column].fillna(mode_value, inplace=True)  # Preenche os valores faltantes com a moda
        else:  # Se for uma coluna numérica
            mean_value = df[column].mean()  # Calcula a média
            df[column].fillna(mean_value, inplace=True)  # Preenche os valores faltantes com a média

    # Salva o DataFrame com os dados completos em um novo arquivo CSV
    df.to_csv(output_file_path, index=False)
    print("Dados completos salvos com sucesso em", output_file_path)

# Caminhos dos arquivos de entrada e saída
input_file_path = 'D:/student/student_data.csv'
output_file_path = 'D:/student/student_data_completed.csv'

# Executa a função para completar os dados faltantes e salvar o novo arquivo CSV
complete_missing_data(input_file_path, output_file_path)