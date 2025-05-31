#!/bin/bash
#SBATCH --job-name=pipeline_end2end      # create a short name for your job
#SBATCH --nodes=1                
#SBATCH --ntasks-per-node=1      
#SBATCH --time=00:30:00          # total run time limit (HH:MM:SS)
#SBATCH --account=F202500001HPCVLABEPICUREx
#SBATCH --partition=normal-x86
#SBATCH --output=pipeline.out  # std out
#SBATCH --error=pipeline.err   # std err

source /share/env/module_select.sh

module purge
module load "Java/17.0.6"


source /projects/F202500001HPCVLABEPICURE/mca57876/ADGD_/env-spark/bin/activate

# Verificações

export SPARK_HOME="/projects/F202500001HPCVLABEPICURE/mca57876/ADGD_/spark-3.5.5-bin-hadoop3"
export PATH="${SPARK_HOME}/bin:${PATH}"

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

echo "MASTER_ADDR: $MASTER_ADDR"
echo "MASTER_PORT: $MASTER_PORT"
echo "WORLD_SIZE: $WORLD_SIZE"
echo "SLURM_RANK: $SLURM_PROCID"
echo "SLURM_LOCALID: $SLURM_LOCALID"

# Configura alguns níveis de log para auxiliar no debug (opcional)
export LOGLEVEL=INFO
export TORCH_CPP_LOG_LEVEL=INFO
export TORCH_DISTRIBUTED_DEBUG=DETAIL

#############################################
# ETAPA 1: (Opcional) Processamento de Dados com Spark
#############################################
#echo "[Pipeline] Spark -> Iniciando o processamento dos dados"
#$SPARK_HOME/bin/spark-submit \
#    --master spark://$head_node_ip:7077 \
#    --num-executors 10 \
#    --executor-cores 10 \
#    /projects/F202500001HPCVLABEPICURE/mca57876/ADGD_/ADGD/src/spark/data_processing.py
#
#if [ $? -ne 0 ]; then
#    echo "[Pipeline] Spark -> Erro no processamento. Encerrando pipeline."
#    exit 1
#fi
#
#echo "[Pipeline] Spark -> Processamento de dados concluído com sucesso."

#############################################
# ETAPA 2: Treinamento Distribuído com PyTorch
#############################################
echo "[Pipeline] Torch -> Iniciando o treinamento distribuído"

srun --nodes=1 --ntasks=1 torchrun \
    --nnodes=$SLURM_NNODES \                         # =10
    --nproc_per_node=$SLURM_NTASKS_PER_NODE \        # =10
    --rdzv_backend=c10d \
    --rdzv_id=$SLURM_JOB_ID \                        # garante ID único p/ este job
    --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} \
    /projects/F202500001HPCVLABEPICURE/mca57876/ADGD_/ADGD/src/torch/data_learning.py \
        --parquet_path /projects/F202500001HPCVLABEPICURE/mca57876/ADGD_/ADGD/outputs/job_logs \
        --epochs 10 \
        --batch_size 128 \
        --learning_rate 0.001


# Verifica se o treinamento foi concluído com sucesso
if [ $? -ne 0 ]; then
    echo "[Pipeline] Torch -> Erro no treinamento. Encerrando pipeline."
    exit 1
fi

echo "[Pipeline] Torch -> Treinamento distribuído concluído com sucesso."
echo "[Pipeline] Pipeline end-to-end executado com êxito."
