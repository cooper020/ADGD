#!/bin/bash
#SBATCH --job-name=spark-cluster # create a short name for your job
#SBATCH --nodes=3 # node count
#SBATCH --cpus-per-task=48 # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=28G # memory per node
#SBATCH --time=00:59:00 # total run time limit (HH:MM:SS)
#SBATCH --account=F202500001HPCVLABEPICUREa #add an a in the end for the arm nodes
#SBATCH --partition=normal-arm
#SBATCH --output=start_cluster.out  # std out
#SBATCH --error=start_cluster.err   # std err

source /share/env/module_select.sh

module purge
module load "Python/3.12.3-GCCcore-13.3.0"
module load "Java/17.0.6"
module load "OpenMPI/5.0.3-GCC-13.3.0"

source /projects/F202500001HPCVLABEPICURE/mca57876/ADGD/env-spark/bin/activate

export PYTHONPATH=/projects/F202500001HPCVLABEPICURE/mca57876/ADGD/env-spark/bin/python
export PYSPARK_PYTHON=/projects/F202500001HPCVLABEPICURE/mca57876/ADGD/env-spark/bin/python
export PYSPARK_DRIVER_PYTHON=/projects/F202500001HPCVLABEPICURE/mca57876/ADGD/env-spark/bin/python
export SPARK_HOME=/projects/F202500001HPCVLABEPICURE/mca57876/ADGD/spark-3.5.5-bin-hadoop3
export PATH="${SPARK_HOME}/bin:${PATH}"

# Get all worker nodes and write to the config file of Spark
scontrol show hostnames | sed -n '2,$p' > $SPARK_HOME/conf/workers

# Start the spark cluster
$SPARK_HOME/sbin/start-all.sh
sleep infinity
 