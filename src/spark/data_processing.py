#!/usr/bin/env python
# coding: utf-8
import sys

import findspark
import pyspark
from pyspark import StorageLevel
from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.sql.functions import rand
from pyspark.sql.functions import size
import argparse
import calendar
import os
import time
from datetime import date, timedelta, datetime, time
from collections import defaultdict

from pyspark.sql.functions import udf
from pyspark.sql.types import ArrayType, StringType
from pyspark.sql.functions import broadcast

# Define a função que expande os nodes
def expand_all_nodes(nodes_str):
    if not nodes_str or nodes_str.strip() == "":
        return []

    import re
    result = []
    parts = re.split(r',(?![^\[]*\])', nodes_str)

    for part in parts:
        match = re.match(r'([a-zA-Z0-9_\-]+)(?:\[(.*?)\])?$', part.strip())
        if not match:
            continue

        prefix, inside = match.groups()

        if inside:
            segments = inside.split(',')
            for seg in segments:
                if '-' in seg:
                    start, end = map(int, seg.split('-'))
                    width = max(len(seg.split('-')[0]), len(seg.split('-')[1]))
                    for i in range(start, end + 1):
                        result.append(f"{prefix}{str(i).zfill(width)}")
                else:
                    result.append(f"{prefix}{seg.zfill(len(seg))}")
        else:
            result.append(prefix)

    return result

# Registar UDF
expand_all_nodes_udf = udf(expand_all_nodes, ArrayType(StringType()))

# Vai bsucar a instalação do spark
findspark.init()

if __name__ == '__main__':

    # 1. Definir argumentos para personalizar por exemplo o período de análise
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--outfile", nargs='?', help="outfile")
    parser.add_argument("-p", "--parquet_path", nargs='?', help="Path to save/load Parquet files", default="/projects/F202500001HPCVLABEPICURE/mca57876/ADGD_/ADGD/outputs")
    parser.add_argument("--load_parquet", action="store_true", help="Load data from Parquet instead of JSON")
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
    
    logstash_nd = logstash_nd.persist(StorageLevel.MEMORY_AND_DISK)

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
    ).persist(StorageLevel.MEMORY_AND_DISK)

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
    ).persist(StorageLevel.MEMORY_AND_DISK)

    
    # Converter strings para o formato de tempo do spark
    slurm_flattened = slurm_flattened \
        .withColumn("start_time", F.to_timestamp("start")) \
        .withColumn("end_time", F.to_timestamp("end")) \
        .withColumn("nodes_list", expand_all_nodes_udf(F.col("nodes")))

    logstash_flattened = logstash_flattened.withColumn("log_time", F.to_timestamp("timestamp"))
    
    # Reparticionar antes do join para distribuir melhor os dados
    slurm_flattened = slurm_flattened.repartition(100)
    logstash_flattened = logstash_flattened.repartition(100, "host")

    # Juntar os data frames
    job_logs = logstash_flattened.join(
        broadcast(slurm_flattened),
        (F.expr("array_contains(nodes_list, host)")) &
        (logstash_flattened["log_time"] >= slurm_flattened["start_time"]) &
        (logstash_flattened["log_time"] <= slurm_flattened["end_time"]),
        "inner"
    )

    # Libertar memória de DataFrames já usados
    logstash_nd.unpersist()
    slurm_flattened.unpersist()
    logstash_flattened.unpersist()
    del logstash_nd, slurm_flattened, logstash_flattened

    # 6. Conta quantos há de cada
    if slurm_nd is not None:
        params['Total number of jobs'] = slurm_nd.count()
        estados = ["COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL", "PENDING"]
        for estado in estados:
            count = slurm_nd.filter(F.col('_source.state') == estado).count()
            params[f'{estado} jobs'] = count

    slurm_nd.unpersist()
    del slurm_nd

    # Criar coluna total_cpus_job, que nos revela quantos cpus foram utilizados no total naquele job.
    job_logs = job_logs.withColumn("total_cpus_job", F.col("ntasks") * F.col("cpus_per_task"))

    # Eliminar colunas desnecessárias
    cols_to_drop = ["facility", "facility_num", "message", "severity", "syslogtag", "cluster", "derived_ec", "std_in", "std_out", "total_cpus", "cpus_per_task"] 
    job_logs = job_logs.drop(*cols_to_drop)

    if not args.load_parquet:
        # Criar diretório de saída se não existir
        os.makedirs(args.parquet_path, exist_ok=True)

        # Consolidar partições antes de gravar
        job_logs_final = job_logs.repartition(10)  # ou um número pequeno como 10

        # Gravação otimizada do job_logs
        job_logs_final.write.mode("overwrite").parquet(f'{args.parquet_path}/job_logs')

        # Log de diagnóstico
        print(f"Número de partições de job_logs: {job_logs.rdd.getNumPartitions()}")
        print(f"Total de registros em job_logs: {job_logs.count()}")

    # Gravar parâmetros
    outfilename = args.outfile if args.outfile else "params.tex"
    with open(f"{OUTDIR}/{outfilename}", "w+") as wfile:
        for key, value in params.items():
            wfile.write(f"{key}: {value}\n")
    print(f"Resultados escritos em {OUTDIR}/{outfilename}")

    sc.stop()