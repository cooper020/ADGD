#!/usr/bin/env python
# coding: utf-8
import sys

import findspark
import pyspark
from pyspark.sql import SparkSession
import pyspark.sql.functions as F
import argparse
import calendar
import os
import time
from datetime import date, timedelta, datetime, time

# Vai bsucar a instalação do spark
findspark.init()

if __name__ == '__main__':

    # 1. Definir argumentos para personalizar por exemplo o período de análise
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--outfile", nargs='?', help="outfile")
    args = parser.parse_args()

    # 2. Diretoria onde vamos bucar os json
    DATADIR = '/projects/F202500001HPCVLABEPICURE/mca57876/ADGD_/DataSets/PRJ'
    OUTDIR = '/projects/F202500001HPCVLABEPICURE/mca57876/ADGD_/ADGD/outputs'

    # 3. Parâmetros para uma contagem do output 
    params = {
        'Total number of jobs': 0,

        'COMPLETED jobs': 0,
        'FAILED jobs': 0,
        'CANCELLED jobs': 0,
        'TIMEOUT jobs': 0,
        'OUT_OF_MEMORY jobs': 0,
        'NODE_FAIL jobs': 0,
        'PENDING jobs': 0
    }

    # 4. Criar a sessão spark para processar tudo paralelamente
    print("FIND SPARK")
    print(findspark.find())
    sc = SparkSession.builder.master("local[*]").appName('Spark-ADGD').getOrCreate()

    # 5. Carregar os dados para um dataframe
    slurm_nd = sc.read.json(f'{DATADIR}/slurm.json', multiLine=True)
    
    logstash_nd = None
    for root, dirs, files in os.walk(DATADIR):
        for f in files:
            if f.startswith('logstash'):
                data = sc.read.json(f'{DATADIR}/{f}', multiLine=True)
                data = data.drop('_ignored')
                if logstash_nd is None:
                    logstash_nd = data
                else:
                    logstash_nd = logstash_nd.union(data)
    
    # Pegar só no que está na coluna source
    slurm_flattened = slurm_nd.select(
        F.col("_source.@end").alias("end"),
        F.col("_source.@queue_wait").alias("queue_wait"),
        F.col("_source.@start").alias("start"),
        F.col("_source.@submit").alias("submit"),
        F.col("_source.cluster").alias("cluster"),
        F.col("_source.cpu_hours").alias("cpu_hours"),
        F.col("_source.cpus_per_task").alias("cpus_per_task"),
        F.col("_source.derived_ec").alias("derived_ec"),
        F.col("_source.elapsed").alias("elapsed"),
        F.col("_source.exit_code").alias("exit_code"),
        F.col("_source.jobid").alias("jobid"),
        F.col("_source.nodes").alias("nodes"),
        F.col("_source.ntasks").alias("ntasks"),
        F.col("_source.partition").alias("partition"),
        F.col("_source.qos").alias("qos"),
        F.col("_source.state").alias("state"),
        F.col("_source.std_in").alias("std_in"),
        F.col("_source.std_out").alias("std_out"),
        F.col("_source.time_limit").alias("time_limit"),
        F.col("_source.total_cpus").alias("total_cpus"),
        F.col("_source.total_nodes").alias("total_nodes")
    )
    
    # Pegar só no que está na coluna source
    logstash_flattened = logstash_nd.select(
        F.col("_source.@timestamp").alias("timestamp"),
        F.col("_source.facility").alias("facility"),
        F.col("_source.facility-num").alias("facility_num"),
        F.col("_source.host").alias("host"),
        F.col("_source.message").alias("message"),
        F.col("_source.severity").alias("severity"),
        F.col("_source.severity-num").alias("severity_num"),
        F.col("_source.syslogtag").alias("syslogtag")
    )
    
    # Converter strings para o formato de tempo do spark
    slurm_flattened = slurm_flattened.withColumn("start_time", F.to_timestamp(F.col("start")))
    slurm_flattened = slurm_flattened.withColumn("end_time", F.to_timestamp(F.col("end")))
    logstash_flattened = logstash_flattened.withColumn("log_time", F.to_timestamp(F.col("timestamp")))
    
    # Criar e explodir uma nova coluna com a lista de nodes usados num job
    slurm_flattened = slurm_flattened.withColumn("nodes_list", F.split(F.col("nodes"), ","))
    slurm_exploded = slurm_flattened.withColumn("node", F.explode_outer("nodes_list"))
    
    # Juntar os data frames
    job_logs = logstash_flattened.join(
        slurm_exploded,
        (logstash_flattened["host"] == slurm_exploded["node"]) &
        (logstash_flattened["log_time"] >= slurm_exploded["start_time"]) &
        (logstash_flattened["log_time"] <= slurm_exploded["end_time"]),
        "inner"
    )
    
    # Mostrar o esquema e os primeiros registros
    job_logs.printSchema()
    job_logs.show(10, truncate=False)
    

    # 6. Conta quantos há de cada
    if slurm_nd is not None:
        params['Total number of jobs'] = slurm_nd.count()
        estados = ["COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL", "PENDING"]
        for estado in estados:
            count = slurm_nd.filter(F.col('_source.state') == estado).count()
            params[f'{estado} jobs'] = count

    # Eliminar colunas desnecessárias
    cols_to_drop = ["facility", "facility_num", "message", "severity", "syslogtag", "cluster", "derived_ec", "std_in", "std_out"] 
    job_logs = job_logs.drop(*cols_to_drop)
    job_logs.printSchema()

    

    outfilename = args.outfile if args.outfile else "params.tex"
    with open(f"{OUTDIR}/{outfilename}", "w+") as wfile:
        for key, value in params.items():
            wfile.write(f"{key}: {value}\n")


    print(f"Resultados escritos em {OUTDIR}/{outfilename}")

    sc.stop()





