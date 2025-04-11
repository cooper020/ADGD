#!/bin/bash
#SBATCH --job-name=spark-pi      # create a short name for your job
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --time=00:55:00          # total run time limit (HH:MM:SS)
#SBATCH --account=F202500001HPCVLABEPICUREa
#SBATCH --partition=normal-arm
#SBATCH --output=start_client.out  # std out
#SBATCH --error=start_client.err   # std err

source /share/env/module_select.sh

module purge
module load "Java/17.0.6"
module load "Python/3.12.3-GCCcore-13.3.0"

# This is basically the same steps as seen in the first subchapter

# Remember to change this to the actual path
export SPARK_HOME="/projects/F202500001HPCVLABEPICURE/mca57876/ADGD_/spark-3.5.5-bin-hadoop3"

# Add to the global PATH
export PATH="${SPARK_HOME}/bin:${PATH}"


# Get the master node, which will be the first node from the command `scontrol show hostnames`
master_node="cna0563"

# Submit job to the Spark Cluster using spark-submit
# In this script we submit a provided example in the spark directory
$SPARK_HOME/bin/spark-submit \
    --master spark://${master_node}:7077 \
    /projects/F202500001HPCVLABEPICURE/mca57876/ADGD_/ADGD/src/spark/data_processing.py