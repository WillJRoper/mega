#!/bin/bash
#SBATCH -A dp004
#SBATCH -p cosma6
#SBATCH --job-name=MEGA-Test
#SBATCH -t 0-12:00
#SBATCH --ntasks 16
# #SBATCH --cpus-per-task=16
# #SBATCH --ntasks-per-node=16
#SBATCH -o logs/out_std_halo.%J
#SBATCH --exclusive

module purge
module load pythonconda3/4.5.4 gnu_comp/7.3.0 openmpi/3.0.1 hdf5/1.10.3

source activate mega-env

EXEC_DIR="./core"
PARM_DIR="./params"

i=$(($SLURM_ARRAY_TASK_ID - 1))

mpiexec -np 16 python $EXEC_DIR/mainMEGA.py $PARM_DIR/mega-param_mpitest.yml $i

echo "Job done, info follows..."
sacct -j $SLURM_JOBID --format=JobID,JobName,Partition,MaxRSS,Elapsed,ExitCode
exit



