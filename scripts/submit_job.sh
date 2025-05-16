#!/bin/bash
#SBATCH --job-name=pipeline_end2end      # create a short name for your job
#SBATCH --nodes=10                
#SBATCH --ntasks-per-node=10      
#SBATCH --time=00:30:00          # total run time limit (HH:MM:SS)
#SBATCH --account=F202500001HPCVLABEPICUREa
#SBATCH --partition=normal-arm
#SBATCH --output=pipeline.out  # std out
#SBATCH --error=pipeline.err   # std err

source /share/env/module_select.sh

module purge
module load "Java/17.0.6"
module load "Python/3.12.3-GCCcore-13.3.0"

source /projects/F202500001HPCVLABEPICURE/mca57876/ADGD_/env-spark/bin/activate

export LD_LIBRARY_PATH="/projects/F202500001HPCVLABEPICURE/mca57876/ADGD_/env-spark/lib64/python3.6/site-packages/torch/lib:$LD_LIBRARY_PATH"

# This is basically the same steps as seen in the first subchapter

# Remember to change this to the actual path
export SPARK_HOME="/projects/F202500001HPCVLABEPICURE/mca57876/ADGD_/spark-3.5.5-bin-hadoop3"

# Add to the global PATHc
export PATH="${SPARK_HOME}/bin:${PATH}"

# Defina o nó mestre para o Spark (use o comando “scontrol show hostnames” se optar por capturar dinamicamente)
master_node="cna0563"

# -------------------------------
# Etapa 1: Processamento de Dados com Spark
# -------------------------------
echo "[Pipeline] Iniciando o processamento dos dados com Spark..."
$SPARK_HOME/bin/spark-submit \
    --master spark://${master_node}:7077 \
    --num-executors 10 \
    --executor-cores 10 \
    /projects/F202500001HPCVLABEPICURE/mca57876/ADGD_/ADGD/src/spark/data_processing.py

# Verifica se o Spark job foi concluído com sucesso
if [ $? -ne 0 ]; then
    echo "[Pipeline] Erro no processamento com Spark. Encerrando pipeline."
    exit 1
fi

echo "[Pipeline] Processamento de dados concluído com sucesso."
# Nota: Certifique-se de que o script Spark gera os arquivos Parquet no diretório esperado,
# por exemplo: /projects/F202500001HPCVLABEPICURE/mca57876/ADGD_/ADGD/outputs/job_logs

# -------------------------------
# Etapa 2: Treinamento Distribuído com PyTorch
# -------------------------------
echo "[Pipeline] Iniciando o treinamento distribuído com PyTorch..."

# Neste exemplo, usamos torchrun para iniciar o treinamento distribuído.
# O argumento --nnodes define o número de nós e --nproc_per_node o número de processos por nó.
# Esses valores devem bater com as opções do SLURM.
torchrun --nnodes=10 --nproc_per_node=10 \
    /projects/F202500001HPCVLABEPICURE/mca57876/ADGD_/ADGD/src/torch/data_learning.py \
    --parquet_path /projects/F202500001HPCVLABEPICURE/mca57876/ADGD_/ADGD/outputs/job_logs \
    --epochs 10 \
    --batch_size 128 \
    --learning_rate 0.001

# Verifica se o treinamento foi concluído com sucesso
if [ $? -ne 0 ]; then
    echo "[Pipeline] Erro no treinamento com PyTorch. Encerrando pipeline."
    exit 1
fi

echo "[Pipeline] Treinamento distribuído concluído com sucesso."
echo "[Pipeline] Pipeline end-to-end executado com êxito."