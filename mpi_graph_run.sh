#!/bin/bash
#SBATCH -A dp004
#SBATCH -p cosma6
#SBATCH --job-name=MEGA-Graph-Test
#SBATCH -t 0-12:00
#SBATCH --ntasks 16
#SBATCH -o logs/out_std_graph.%J
#SBATCH -e logs/err_std_graph.%J
#SBATCH --exclusive

module purge
module load pythonconda3/4.5.4 gnu_comp/7.3.0 openmpi/3.0.1 hdf5/1.10.3

source activate mega-env

EXEC_DIR="./core"
PARM_DIR="./params"

i=$(($SLURM_ARRAY_TASK_ID - 1))

for i in {0..61}
do
    mpiexec -np 16 python $EXEC_DIR/mainMEGA.py $PARM_DIR/mega-param_graph_mpitest.yml $i
done
echo "Job done, info follows..."
sacct -j $SLURM_JOBID --format=JobID,JobName,Partition,MaxRSS,Elapsed,ExitCode
exit

