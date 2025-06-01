#!/bin/bash
#SBATCH --job-name=pipeline_end2end
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=dev-x86
#SBATCH --account=F202500001HPCVLABEPICUREx
#SBATCH --time=00:20:00
#SBATCH --output=pipeline_dev_x86.out
#SBATCH --error=pipeline_dev_x86.err

pipeline_start=$(date +%s)

source /share/env/module_select.sh
module purge
module spider Python  
module load "Python/3.12.3-GCCcore-13.3.0"
module load "Java/17.0.6"

source /projects/F202500001HPCVLABEPICURE/mca57876/ADGD_/env-spark/bin/activate

python -m pip install --upgrade pip setuptools wheel
python -m pip install torch torchvision  
python -m pip install pandas scikit-learn tqdm   
python -m pip install pyarrow fastparquet matplotlib seaborn
python -m pip install findspark
python -m pip install pyspark

echo "SJN: $SLURM_JOB_NODELIST"

nodes=($(scontrol -a show hostnames ${SLURM_JOB_NODELIST} | sort | uniq))
echo "Nodes: ${nodes[@]}"

head_node_ip=$(srun --nodes=1 --ntasks=1 hostname --ip-address | uniq)
echo "Head node IP: $head_node_ip"

export WORLD_SIZE=$SLURM_NTASKS
export RANK=$SLURM_PROCID
export LOCAL_RANK=$SLURM_LOCALID

export MASTER_ADDR=$head_node_ip
export MASTER_PORT=29500

export SPARK_HOME="/projects/F202500001HPCVLABEPICURE/mca57876/ADGD_/spark-3.5.5-bin-hadoop3"
export PATH="${SPARK_HOME}/bin:${PATH}"

echo "Running Spark..."
spark_start=$(date +%s)
$SPARK_HOME/bin/spark-submit \
    --master spark://$head_node_ip:7077 \
    --num-executors 128 \
    --executor-cores 1 \
    /projects/F202500001HPCVLABEPICURE/mca57876/ADGD_/ADGD/src/spark/data_processing.py

if [ $? -ne 0 ]; then
    echo "[Pipeline] Spark -> Erro no processamento. Encerrando pipeline."
    exit 1
fi
spark_end=$(date +%s)
echo "Spark runtime: $((spark_end - spark_start)) seconds"

echo "Running Torch..."
torch_start=$(date +%s)
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
torch_end=$(date +%s)
echo "Torch runtime: $((torch_end - torch_start)) seconds"   

echo "Running Inference..."
inference_start=$(date +%s)
python /projects/F202500001HPCVLABEPICURE/mca57876/ADGD_/ADGD/src/torch/inference.py \
   --input_dir "/projects/F202500001HPCVLABEPICURE/mca57876/ADGD_/ADGD/outputs/job_logs" \
   --model_path "/projects/F202500001HPCVLABEPICURE/mca57876/ADGD_/ADGD/outputs/pytorch_model.pt" \
   --scaler_path "/projects/F202500001HPCVLABEPICURE/mca57876/ADGD_/ADGD/outputs/scaler.pkl" \
   --output_path "/projects/F202500001HPCVLABEPICURE/mca57876/ADGD_/ADGD/outputs/predictions.csv"
inference_end=$(date +%s)
echo "Inference runtime: $((inference_end - inference_start)) seconds"     

pipeline_end=$(date +%s)
echo "Pipeline total runtime: $((pipeline_end - pipeline_start)) seconds"