#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=16G
#SBATCH --time=0-2:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --job-name train
#SBATCH --open-mode=append
#SBATCH --output /gpfs/scratch/changd14/tiny_imagenet/output/slurm_output/train-%J.log 

module load singularity/3.7.1
singularity exec --bind /gpfs/scratch --nv --overlay /gpfs/scratch/changd14/overlay-25GB-500K.ext3:ro /gpfs/scratch/changd14/cuda11.7.99-cudnn8.5-devel-ubuntu22.04.2.sif /bin/bash -c "
source /ext3/miniconda3/etc/profile.d/conda.sh
export HF_HOME=/gpfs/scratch/changd14/.cache
export PATH=/ext3/miniconda3/bin:$PATH
conda activate 2590-hw3
cd /gpfs/scratch/changd14/tiny_imagenet
python train.py --depth $1 --width_factor $2
"