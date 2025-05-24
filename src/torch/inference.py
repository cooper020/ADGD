#!/usr/bin/env python
# coding: utf-8

import argparse
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

class MLPModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, output_dim=7):
        super(MLPModel, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        return self.net(x)

def main():
    parser = argparse.ArgumentParser(description="Realiza inferência usando o modelo treinado")
    parser.add_argument("--input_path", type=str, default="inference_input.parquet",
                        help="Caminho para os dados de entrada (em formato Parquet)")
    parser.add_argument("--model_path", type=str, default="pytorch_model.pt",
                        help="Caminho para o modelo treinado")
    parser.add_argument("--output_path", type=str, default="predictions.csv",
                        help="Caminho para salvar as previsões")
    args = parser.parse_args()

    try:
        df = pd.read_parquet(args.input_path)
    except Exception as e:
        print("Erro ao carregar os dados para inferência:", e)
        return

    #  Verificar se não se perderam features
    features = ["elapsed", "ntasks", "time_limit", "total_nodes", "total_cpus_job"]
    if not all(feature in df.columns for feature in features):
        print("Os dados de entrada não possuem todos os features necessários.")
        return

    X = df[features].values

    # É importante aplicar a mesma normalização utilizada no treinamento
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X) 

    # Define e carrega o modelo treinado
    input_dim = X.shape[1]
    model = MLPModel(input_dim=input_dim)
    try:
        state_dict = torch.load(args.model_path, map_location=torch.device("cpu"))
        model.load_state_dict(state_dict)
    except Exception as e:
        print("Erro ao carregar o modelo:", e)
        return
    
    model.eval()
    with torch.no_grad():
        inputs = torch.tensor(X_scaled, dtype=torch.float32)
        outputs = model(inputs)
        _, predictions = torch.max(outputs, dim=1)

    # Salva as previsões
    df["prediction"] = predictions.numpy()
    df.to_csv(args.output_path, index=False)
    print(f"Previsões salvas em {args.output_path}")

if __name__ == "__main__":
    main()
