#!/usr/bin/env python
# coding: utf-8

import os
import argparse
import pandas as pd
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# Dataset customizado que recebe os dados já pré-processados (em arrays)
class JobLogsDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# Modelo simples; pode ser ajustado (ex.: mais camadas, dropout, etc.)
class MLPModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, output_dim=7):
        """
        input_dim: número de features de entrada
        hidden_dim: tamanho da camada oculta (pode ser ajustado)
        output_dim: número de classes (neste caso, 7 estados de job)
        """
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
    # Parser de argumentos
    parser = argparse.ArgumentParser(description="PyTorch Distributed Training para previsão de falhas de job")
    parser.add_argument("--parquet_path", type=str,
                        default="/projects/F202500001HPCVLABEPICURE/mca57876/ADGD_/ADGD/outputs/job_logs",
                        help="Caminho para o diretório com o arquivo Parquet (job_logs)")
    parser.add_argument("--epochs", type=int, default=10, help="Número de épocas de treinamento")
    parser.add_argument("--batch_size", type=int, default=128, help="Tamanho do batch")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Taxa de aprendizado")
    # Parâmetro para integração com treinamento distribuído (usado pelo launcher do PyTorch)
    parser.add_argument("--local_rank", type=int, default=0, help="Rank local do processo")
    args = parser.parse_args()

    # Inicializar o grupo distribuído (usa variáveis de ambiente, geralmente configuradas pelo launcher)
    dist.init_process_group(backend="gloo", init_method="env://")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.manual_seed(42)
    device = torch.device("cuda:{}".format(args.local_rank) if torch.cuda.is_available() else "cpu")
    
    if rank == 0:
        print(f"[Rank {rank}] Iniciando o treinamento distribuído com {world_size} processos")
        print("Carregando os dados a partir de:", args.parquet_path)
    
    # Carregar os dados do Parquet usando pandas
    try:
        df = pd.read_parquet(args.parquet_path)
    except Exception as e:
        print("Erro ao carregar o arquivo Parquet:", e)
        return

    # Mapeamento dos estados para rótulos numéricos
    state_mapping = {
        "COMPLETED": 0,
        "FAILED": 1,
        "CANCELLED": 2,
        "TIMEOUT": 3,
        "OUT_OF_MEMORY": 4,
        "NODE_FAIL": 5,
        "PENDING": 6
    }
    
    # Certifique-se de que as colunas necessárias existem no dataframe
    required_features = ["elapsed", "ntasks", "time_limit", "total_nodes", "total_cpus_job", "state"]
    df = df.dropna(subset=required_features)
    
    # Aplicar o mapeamento para a coluna 'state'
    df["state_label"] = df["state"].map(state_mapping)
    df = df.dropna(subset=["state_label"])
    
    # Selecionar as features e o alvo
    features = ["elapsed", "ntasks", "time_limit", "total_nodes", "total_cpus_job"]
    X = df[features].values
    y = df["state_label"].values.astype(int)
    
    # Dividir os dados em treinamento e validação (80% / 20%)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Normalizar as features para ajudar no treinamento
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    
    # Criar datasets e dataloaders
    train_dataset = JobLogsDataset(X_train, y_train)
    val_dataset = JobLogsDataset(X_val, y_val)
    
    # Usar DistributedSampler para o dataset de treinamento (garante que cada processo veja uma fração dos dados)
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, sampler=train_sampler, drop_last=True
    )
    # Para validação não é necessário usar sampler
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)
    
    # Instanciar o modelo, enviá-lo para o dispositivo e encapsulá-lo com DDP
    input_dim = X_train.shape[1]
    model = MLPModel(input_dim=input_dim, hidden_dim=64, output_dim=7)
    model.to(device)
    model = DDP(model, device_ids=[args.local_rank] if torch.cuda.is_available() else None)
    
    # Definir a função perda e o otimizador
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    
    # Loop de treinamento
    for epoch in range(args.epochs):
        model.train()
        # Atualiza o sampler para garantir a aleatoriedade entre épocas
        train_sampler.set_epoch(epoch)
        epoch_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            # Acumula a perda e acertos
            epoch_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        # Como cada processo vê uma parte dos dados, agregamos a perda com all_reduce
        loss_tensor = torch.tensor(epoch_loss, device=device)
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
        avg_loss = loss_tensor.item() / len(train_dataset)        
        train_acc = 100.0 * correct / total
        
        if rank == 0:
            print(f"Epoch [{epoch+1}/{args.epochs}] - Treinamento: Loss: {avg_loss:.4f}, Accuracy: {train_acc:.2f}%")
        
        # Validação (pode ser feita sem sincronização, pois é só para monitoramento)
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        with torch.no_grad():
            for inputs_val, labels_val in val_loader:
                inputs_val, labels_val = inputs_val.to(device), labels_val.to(device)
                outputs_val = model(inputs_val)
                loss_val = criterion(outputs_val, labels_val)
                val_loss += loss_val.item() * inputs_val.size(0)
                _, predicted_val = torch.max(outputs_val, dim=1)
                total_val += labels_val.size(0)
                correct_val += (predicted_val == labels_val).sum().item()
        val_loss = val_loss / len(val_dataset)
        val_acc = 100.0 * correct_val / total_val
        if rank == 0:
            print(f"           Validação:   Loss: {val_loss:.4f}, Accuracy: {val_acc:.2f}%")
    
    # Salvar o modelo apenas pelo processo principal (rank 0)
    if rank == 0:
        model_path = "pytorch_model.pt"
        # Salvar apenas os parâmetros do módulo encapsulado (sem a camada DDP)
        torch.save(model.module.state_dict(), model_path)
        print("Modelo salvo em", model_path)
    
    # Encerra o grupo distribuído
    dist.destroy_process_group()

if __name__ == "__main__":
    main()
