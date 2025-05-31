#!/bin/bash
#SBATCH --job-name=dev-arm
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=dev-arm
#SBATCH --account=F202500001HPCVLABEPICUREa
#SBATCH --time=00:20:00
#SBATCH --output=dev-arm.out
#SBATCH --error=dev-arm.err

# 0) Carrega Lmod e “limpa” módulos anteriores
source /share/env/module_select.sh
module purge

# 1) Descobre qual Python ARM está disponível
#    Você verá algo como Python/3.6.8 ou Python/3.9.4
module spider Python  
#    Após ver a lista, carregue a versão que existir, por ex:
module load "Python/3.12.3-GCCcore-13.3.0"

echo ">>> Python carregado: $(which python) ($(python --version))"
module list

# 2) Cria (ou ativa) um venv usando exatamente esse Python
VENV_DIR=$HOME/venv/arm_torch
if [ ! -d "$VENV_DIR" ]; then
  python -m venv "$VENV_DIR"
fi
source "$VENV_DIR/bin/activate"
echo ">>> Virtualenv ativo: $(which python) ($(python --version))"

# 3) Atualiza pip e instala dependências via python -m pip
python -m pip install --upgrade pip setuptools wheel

# 4) Instala PyTorch para ARM64
#    Ajuste este índice caso precise de outra build ou versão
python -m pip install torch torchvision \
   --extra-index-url https://download.pytorch.org/whl/arm64

python -m pip install pandas scikit-learn tqdm   

python -m pip install pyarrow fastparquet


# 5) Valida import e versão
echo "=== DEBUG PYTORCH ==="
python - << 'EOF'
import torch
print("Torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
EOF
echo "====================="


echo "Cenas de exports"
# Exibe o SLURM_JOB_NODELIST para depuração
echo "SJN: $SLURM_JOB_NODELIST"

# Obtém a lista única de nós alocados
nodes=($(scontrol -a show hostnames ${SLURM_JOB_NODELIST} | sort | uniq))
echo "Nodes: ${nodes[@]}"

# Obtém o IP do nó principal (head node) utilizando srun
head_node_ip=$(srun --nodes=1 --ntasks=1 hostname --ip-address | uniq)
echo "Head node IP: $head_node_ip"

# Define as variáveis de ambiente para o treinamento distribuído
export WORLD_SIZE=$SLURM_NTASKS
#  export RANK=$SLURM_PROCID
#  export LOCAL_RANK=$SLURM_LOCALID

export MASTER_ADDR=$head_node_ip
export MASTER_PORT=29500

# 6) Chama seu script de treino (single‐process ou DDP conforme preferência)
#    Aqui rodamos em modo single‐process. Para DDP, use torchrun conforme vimos.
srun --nodes=1 --ntasks=1 torchrun \
    --nnodes=$SLURM_NNODES \
    --nproc_per_node=$SLURM_NTASKS_PER_NODE \
    --rdzv_backend=c10d \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} \
    /projects/F202500001HPCVLABEPICURE/mca57876/ADGD_/ADGD/src/torch/data_learning.py \
    --parquet_path /projects/F202500001HPCVLABEPICURE/mca57876/ADGD_/ADGD/outputs/job_logs \
    --epochs 10 \
    --batch_size 128 \
    --learning_rate 0.001
