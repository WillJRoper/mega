#!/bin/bash
#SBATCH -A dp004
#SBATCH -p cosma6
#SBATCH --job-name=MEGA-Test
#SBATCH -t 0-12:00
#SBATCH --ntasks 1
#SBATCH --array=1-62
#SBATCH --cpus-per-task=16
#SBATCH --ntasks-per-node=1
#SBATCH -o logs/out_std_%N.%J
#SBATCH -e logs/err_std_%N.%J
#SBATCH --exclusive

module purge
module load pythonconda3/4.5.4

unset I_MPI_HYDRA_BOOTSTRAP
export I_MPI_ADJUST_ALLGATHER=1

export I_MPI_DAPL_CHECK_MAX_RDMA_SIZE=enable
export I_MPI_DAPL_MAX_MSG_SIZE=1073741824

EXEC_DIR="./core"
PARM_DIR="./params"

i=$(($SLURM_ARRAY_TASK_ID - 1))

$MPIROOT/bin64/mpirun -np 16 python $EXEC_DIR/mainMEGA.py $PARM_DIR/mega-param_mpitest.yml $i

echo "Job done, info follows..."
sacct -j $SLURM_JOBID --format=JobID,JobName,Partition,MaxRSS,Elapsed,ExitCode
exit

