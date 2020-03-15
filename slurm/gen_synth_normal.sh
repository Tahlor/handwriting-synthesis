#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH -C 'rhel7&pascal'
#SBATCH --mem-per-cpu 6000
#SBATCH --ntasks 8
#SBATCH --time 36:15:00
#SBATCH --mail-user=taylornarchibald@gmail.com   # email address
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --output="/panfs/pan.fsl.byu.edu/scr/grp/fslg_hwr/handwriting-synthesis/slurm/log_gen_synth_normal.slurm"


#%Module

module purge
module load cuda/10.1
module load cudnn/7.6

export PATH="/fslhome/tarch/.conda/envs/tf16:$PATH"
eval "$(conda shell.bash hook)"
conda activate /fslhome/tarch/.conda/envs/tf16

cd "/panfs/pan.fsl.byu.edu/scr/grp/fslg_hwr/handwriting-synthesis"
which python
#python -u demo.py
python -u create_synthetic_training_data.py --variant "normal"
