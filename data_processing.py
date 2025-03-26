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

    # 5. Todos os dados para um dataframe novo nd
    nd = None
    for root, dirs, files in os.walk(DATADIR):
        for f in files:
            if f.startswith('slurm') and f.endswith('.json'):
                data = sc.read.json(f'{DATADIR}/{f}', multiLine=True)
                if nd is None:
                    nd = data
                else:
                    nd = nd.union(data)

    # 6. Conta quantos há de cada
    if nd is not None:
        params['Total number of jobs'] = nd.count()
        estados = ["COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL", "PENDING"]
        for estado in estados:
            count = nd.filter(F.col('_source.state') == estado).count()
            params[f'{estado} jobs'] = count

    
    # nd.printSchema()
        
    # 7. Valores diferentes dos atributos na struct src
    selected_data = nd.select(
        F.col('_source.state'),
        F.col('_source.cluster'),
        F.col('_source.partition'),
        F.col('_source.nodes')
    )

    # Mostrar os valores distintos
    selected_data.distinct().show(truncate=False)

    nd = nd.withColumn("cluster",
        F.when(F.col('Partition').contains("arm"), "ARM")
        .otherwise(
            F.when(F.col('Partition').contains("a100"), "GPU")
            .otherwise("AMD")
        )
    )

    
    outfilename = args.outfile if args.outfile else "params.tex"
    with open(f"{OUTDIR}/{outfilename}", "w+") as wfile:
        for key, value in params.items():
            wfile.write(f"{key}: {value}\n")


    print(f"Resultados escritos em {OUTDIR}/{outfilename}")

    sc.stop()



