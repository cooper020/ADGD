#!/usr/bin/env python
# coding: utf-8

import argparse
import glob
import os
import pickle
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

# Dependências para análise e visualização
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# --- Definição do Modelo (deve ser idêntico ao utilizado no treinamento) ---
class MLPModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, output_dim=7):
        super(MLPModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# --- Função de Análise dos Resultados ---
def analyze_results(csv_path):
    """
    Carrega o CSV gerado e analisa as predições.
    Se existir uma coluna 'state' (ground truth), calcula acurácia, relatório
    de classificação e plota matriz de confusão.
    Caso contrário, exibe a distribuição das classes previstas.
    Retorna um dicionário com métricas (ex.: 'accuracy') para uso na conclusão.
    """
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print("Erro ao carregar o CSV para análise:", e)
        return None
    
    results = {}
    print("\n--- Análise de Resultados ---")
    if "state" in df.columns:
        # Mapeamento dos rótulos (o mesmo utilizado no treinamento)
        state_mapping = {
            "COMPLETED": 0,
            "FAILED": 1,
            "CANCELLED": 2,
            "TIMEOUT": 3,
            "OUT_OF_MEMORY": 4,
            "NODE_FAIL": 5,
            "PENDING": 6
        }
        # Converte os rótulos reais para numérico se necessário
        if df["state"].dtype == "object":
            df["state_num"] = df["state"].map(state_mapping)
        else:
            df["state_num"] = df["state"]
        
        y_true = df["state_num"]
        y_pred = df["prediction"]
        acc = accuracy_score(y_true, y_pred)
        results["accuracy"] = acc
        print("Acurácia: {:.2f}%".format(acc * 100))
        print("\nRelatório de Classificação:")
        print(classification_report(y_true, y_pred))
        
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=list(state_mapping.keys()),
                    yticklabels=list(state_mapping.keys()))
        plt.xlabel("Classe Predita")
        plt.ylabel("Classe Verdadeira")
        plt.title("Matriz de Confusão")
        plt.show()
    else:
        print("Não há coluna 'state' com os rótulos reais. Exibindo distribuição das predições.")
        plt.figure(figsize=(8,4))
        df["prediction"].value_counts().sort_index().plot(kind="bar")
        plt.xlabel("Classe Predita")
        plt.ylabel("Frequência")
        plt.title("Distribuição das Predições")
        plt.show()
        results["accuracy"] = None
    return results

# --- Função de Conclusão ---
def conclude_results(results):
    print("\n--- Conclusões Finais ---")
    if results is not None and results.get("accuracy") is not None:
        acc = results["accuracy"]
        print(f"O modelo obteve uma acurácia geral de {acc*100:.2f}%.")
        print("A matriz de confusão e o relatório de classificação indicam que há confusões entre")
        print("determinadas classes, o que sugere a necessidade de aprimoramentos na extração de features")
        print("ou na arquitetura do modelo.")
        print("Recomenda-se, para trabalhos futuros, salvar o scaler utilizado no treinamento para garantir")
        print("que o pré-processamento na inferência seja idêntico e, assim, evitar discrepâncias que possam")
        print("afetar a performance.")
    else:
        print("Como os rótulos reais não foram fornecidos, não é possível concluir detalhadamente sobre a performance.")
        print("Sugere-se coletar ground truth para uma avaliação mais precisa.")
    
# --- Função Main ---
def main():
    parser = argparse.ArgumentParser(
        description="Realiza inferência usando o modelo treinado e analisa os resultados"
    )
    parser.add_argument("--input_dir", type=str, default="inference_data",
                        help="Diretório com os arquivos Parquet de entrada")
    parser.add_argument("--model_path", type=str, default="pytorch_model.pt",
                        help="Caminho para o modelo treinado")
    parser.add_argument("--output_path", type=str, default="predictions.csv",
                        help="Caminho para salvar as previsões")
    parser.add_argument("--scaler_path", type=str, default="scaler.pkl",
                        help="Caminho para o scaler salvo (opcional)")
    args = parser.parse_args()
    
    # Lista todos os arquivos Parquet no diretório informado
    parquet_files = glob.glob(os.path.join(args.input_dir, "*.parquet"))
    if not parquet_files:
        print("Nenhum arquivo Parquet encontrado no diretório:", args.input_dir)
        return
    
    # Lê e concatena os arquivos
    list_df = []
    for file in parquet_files:
        try:
            df_part = pd.read_parquet(file)
            list_df.append(df_part)
        except Exception as e:
            print(f"Erro ao ler {file}: {e}")
    if not list_df:
        print("Falha ao carregar qualquer arquivo Parquet.")
        return
    df = pd.concat(list_df, ignore_index=True)
    
    # Verifica se os features necessários estão presentes
    features = ["elapsed", "ntasks", "time_limit", "total_nodes", "total_cpus_job"]
    if not all(feature in df.columns for feature in features):
        print("Os dados de entrada não possuem todos os features necessários.")
        return
    
    X = df[features].values

    # Pré-processamento: tenta carregar um scaler salvo; se falhar, aplica fit_transform
    try:
        with open(args.scaler_path, "rb") as f:
            scaler = pickle.load(f)
        print("Scaler carregado de", args.scaler_path)
        X_scaled = scaler.transform(X)
    except Exception as e:
        print("Não foi possível carregar o scaler ({}). Usando fit_transform nos dados de inferência.".format(e))
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    
    # Cria o modelo com a mesma arquitetura utilizada no treinamento
    input_dim = X.shape[1]
    model = MLPModel(input_dim=input_dim, hidden_dim=64, output_dim=7)
    
    # Carrega os pesos salvos do modelo
    try:
        state_dict = torch.load(args.model_path, map_location=torch.device("cpu"))
        model.load_state_dict(state_dict)
        print("Modelo carregado de", args.model_path)
    except Exception as e:
        print("Erro ao carregar o modelo:", e)
        return
    
    model.eval()
    with torch.no_grad():
        inputs = torch.tensor(X_scaled, dtype=torch.float32)
        outputs = model(inputs)
        _, predictions = torch.max(outputs, dim=1)
    
    # Acrescenta a coluna "prediction" e salva o CSV
    df["prediction"] = predictions.numpy()
    try:
        df.to_csv(args.output_path, index=False)
        print(f"Previsões salvas em {args.output_path}")
    except Exception as e:
        print("Erro ao salvar as predições:", e)
        return

    # Realiza a análise dos resultados e extrai métricas
    results = analyze_results(args.output_path)
    
    # Imprime conclusões baseadas na análise dos resultados
    conclude_results(results)

if __name__ == "__main__":
    main()
