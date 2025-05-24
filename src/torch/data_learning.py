#!/usr/bin/env python
import os
import time
import datetime
import argparse
import logging

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Supondo que você use tqdm para exibir o progresso
from tqdm import tqdm

# Configuração do logging para debug
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

logger.debug(f"MASTER_ADDR: {os.environ.get('MASTER_ADDR', 'Não definida')}")
logger.debug(f"MASTER_PORT: {os.environ.get('MASTER_PORT', 'Não definida')}")

#####################################
# Funções de Setup e Cleanup para DDP
#####################################
def setup(rank, world_size):
    master_addr = os.environ["MASTER_ADDR"]
    master_port = os.environ["MASTER_PORT"]
    init_method = f"tcp://{master_addr}:{master_port}"
    timeout = datetime.timedelta(seconds=300)
    dist.init_process_group(
        backend="gloo",  # ou "nccl" se estiver usando GPUs
        init_method=init_method,
        rank=rank,
        world_size=world_size,
        timeout=timeout,
    )
    logger.debug(f"Grupo distribuído iniciado: Rank {rank} de {world_size}")

def cleanup():
    dist.destroy_process_group()

#####################################
# Definição do Dataset e do Modelo
#####################################
class JobLogsDataset(Dataset):
    def __init__(self, X, y):
        # Converter os dados para float32 e os labels para int64.
        self.X = X.astype('float32')
        self.y = y.astype('int64')
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return torch.tensor(self.X[idx]), torch.tensor(self.y[idx])

class MLPModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLPModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

#####################################
# Função Principal
#####################################
def main():
    logger.debug("DEBUG :: 1 - Início do processamento distribuído.")

    parser = argparse.ArgumentParser(
        description="Treinamento Distribuído com PyTorch DDP - Data Learning"
    )
    parser.add_argument("--parquet_path", type=str,
                        default="/projects/F202500001HPCVLABEPICURE/mca57876/ADGD_/ADGD/outputs/job_logs",
                        help="Caminho para o arquivo Parquet (job_logs)")
    parser.add_argument("--epochs", type=int, default=5, help="Número de épocas de treinamento")
    parser.add_argument("--batch_size", type=int, default=64, help="Tamanho do batch")
    parser.add_argument("--learning_rate", type=float, default=0.01, help="Taxa de aprendizado")
    parser.add_argument("--local_rank", type=int, default=0, help="Rank local do processo")
    args = parser.parse_args()

    # Obter as variáveis de ambiente para DDP
    try:
        world_size = int(os.environ["WORLD_SIZE"])
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
    except KeyError as e:
        raise EnvironmentError(f"Variável de ambiente indispensável não definida: {e}")

    logger.debug(f"World Size: {world_size}, Rank: {rank}, Local Rank: {local_rank}")

    # Setup do grupo distribuído
    setup(rank, world_size)

    # Define dispositivo (se houver GPU, use-a; caso contrário, CPU)
    device = torch.device(f"cuda:{args.local_rank}" if torch.cuda.is_available() else "cpu")
    logger.debug(f"Utilizando dispositivo: {device}")

    ##############################################
    # Lógica de Data Learning (mantida conforme solicitado)
    ##############################################
    try:
        df = pd.read_parquet(args.parquet_path)
    except Exception as e:
        print("Erro ao carregar o arquivo Parquet:", e)
        cleanup()
        return

    print("DEBUG :: 5")

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
    
    print("DEBUG :: 6")
    # Certifique-se de que as colunas necessárias existem no dataframe
    required_features = ["elapsed", "ntasks", "time_limit", "total_nodes", "total_cpus_job", "state"]
    df = df.dropna(subset=required_features)
    
    print("DEBUG :: 7")
    # Aplicar o mapeamento para a coluna 'state'
    df["state_label"] = df["state"].map(state_mapping)
    df = df.dropna(subset=["state_label"])
    
    # Selecionar as features e o alvo
    features = ["elapsed", "ntasks", "time_limit", "total_nodes", "total_cpus_job"]
    X = df[features].values
    y = df["state_label"].values.astype(int)
    
    # Dividir os dados em treinamento e validação (80% / 20%)
    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Normalizar as features
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    
    # Criar datasets
    train_dataset = JobLogsDataset(X_train, y_train)
    val_dataset = JobLogsDataset(X_val, y_val)
    
    # Configurar DataLoader com DistributedSampler para o conjunto de treinamento
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        drop_last=True
    )
    # Para validação, não é necessário o sampler
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)
    
    # Instanciar o modelo
    input_dim = X_train.shape[1]
    model = MLPModel(input_dim=input_dim, hidden_dim=64, output_dim=7)
    model.to(device)
    # Envolver o modelo com DistributedDataParallel
    model = DDP(model, device_ids=[args.local_rank] if torch.cuda.is_available() else None)
    
    # Definir função perda e otimizador
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    
    ##############################################
    # Loop de Treinamento Distribuído
    ##############################################
    for epoch in range(args.epochs):
        model.train()
        train_sampler.set_epoch(epoch)  # Garante aleatoriedade entre épocas
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
            
            # Acumular loss e acertos
            epoch_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        # Agregar a loss entre os processos
        loss_tensor = torch.tensor(epoch_loss, device=device)
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
        avg_loss = loss_tensor.item() / len(train_dataset)
        train_acc = 100.0 * correct / total
        
        if rank == 0:
            print(f"Epoch [{epoch+1}/{args.epochs}] - Treinamento: Loss: {avg_loss:.4f}, Accuracy: {train_acc:.2f}%")
        
        # Validação
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
        model_path = "/projects/F202500001HPCVLABEPICURE/mca57876/ADGD_/ADGD/outputs/pytorch_model.pt"
        torch.save(model.module.state_dict(), model_path)
        print("Modelo salvo em", model_path)
    
    # Cleanup do grupo distribuído
    cleanup()

if __name__ == "__main__":
    main()
