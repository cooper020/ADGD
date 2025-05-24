from airflow import DAG
from airflow.operators.bash_operator import BashOperator
from datetime import datetime

default_args = {
    'start_date': datetime(2023, 1, 1),  # ajuste a data conforme necessário
}

dag = DAG('pipeline_end_to_end', default_args=default_args, schedule_interval=None, catchup=False)

spark_task = BashOperator(
    task_id='spark_preprocessing',
    bash_command="""
    $SPARK_HOME/bin/spark-submit \
        --master spark://cna0563:7077 \
        --num-executors 10 \
        --executor-cores 10 \
        /projects/F202500001HPCVLABEPICURE/mca57876/ADGD_/ADGD/src/spark/data_processing.py
    """,
    dag=dag
)

torch_train_task = BashOperator(
    task_id='torch_training',
    bash_command="""
    torchrun --nnodes=10 --nproc_per_node=10 \
       /projects/F202500001HPCVLABEPICURE/mca57876/ADGD_/ADGD/src/torch/data_learning.py \
       --parquet_path /projects/F202500001HPCVLABEPICURE/mca57876/ADGD_/ADGD/outputs/job_logs \
       --epochs 10 --batch_size 128 --learning_rate 0.001
    """,
    dag=dag
)

inference_task = BashOperator(
    task_id='inference',
    bash_command="""
    python /projects/F202500001HPCVLABEPICURE/mca57876/ADGD_/ADGD/src/torch/inference.py \
       --input_path /projects/F202500001HPCVLABEPICURE/mca57876/ADGD_/ADGD/outputs/job_logs \
       --model_path pytorch_model.pt --output_path predictions.csv
    """,
    dag=dag
)

# Definindo a ordem das tarefas
spark_task >> torch_train_task >> inference_task
