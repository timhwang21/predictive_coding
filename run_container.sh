#!/bin/bash
#SBATCH --mail-type=ALL
#SBATCH --mail-user=monica.li@uconn.edu
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --exclude=cn[65-69,71-136,153-256,265-320,325-328]
#SBATCH -e error_%A.log
#SBATCH -o output_%A.log
#SBATCH --job-name=PCSWR
##### END OF JOB DEFINITION  #####
export OMP_NUM_THREADS=$SLURM_JOB_CPUS_PER_NODE

# singularity settings
module load singularity/3.1

img_path=/scratch/yil14028/containers/pcswr.simg

# run container
singularity run --bind /scratch:/scratch ${img_path} "$@"
