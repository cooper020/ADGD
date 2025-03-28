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
    DATADIR = '/projects/F202500001HPCVLABEPICURE/mca57876/ADGD/DataSets/PRJ'
    OUTDIR = '/projects/F202500001HPCVLABEPICURE/mca57876/ADGD/TP/outputs'

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
    slurm_nd.printSchema()

    slurm_nd.show(5, truncate=False)
    
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
    logstash_nd.printSchema()
    
    logstash_nd.show(5, truncate=False)

    # 6. Conta quantos há de cada
    if slurm_nd is not None:
        params['Total number of jobs'] = slurm_nd.count()
        estados = ["COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL", "PENDING"]
        for estado in estados:
            count = slurm_nd.filter(F.col('_source.state') == estado).count()
            params[f'{estado} jobs'] = count

    # 7. Valores diferentes dos atributos na struct src
    selected_data = slurm_nd.select(
        F.col('_source.state'),
        F.col('_source.cluster'),
        F.col('_source.partition'),
        F.col('_source.nodes')
    )

    # Mostrar os valores distintos
    selected_data.distinct().show(truncate=False)

    slurm_nd = slurm_nd.withColumn("cluster",
        F.when(F.col('_source.partition').contains("arm"), "ARM")
        .otherwise(
            F.when(F.col('_source.partition').contains("a100"), "GPU")
            .otherwise("AMD")
        )
    )

    
    outfilename = args.outfile if args.outfile else "params.tex"
    with open(f"{OUTDIR}/{outfilename}", "w+") as wfile:
        for key, value in params.items():
            wfile.write(f"{key}: {value}\n")


    print(f"Resultados escritos em {OUTDIR}/{outfilename}")

    sc.stop()



