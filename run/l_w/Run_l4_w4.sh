#!/bin/sh
#SBATCH --partition=general --qos=short
#SBATCH --time=2:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=2G
#SBATCH --mail-type=ALL
#SBATCH --gres=gpu
##SBATCH --nodelist=ewi1
##SBATCH --exclude=insy6,insy12
#SBATCH --chdir=/tudelft.net/staff-bulk/ewi/insy/MMC/xinsheng/code/Unitdiscovery/Word-boundary-discovery

module use /opt/insy/modulefiles
module load cuda/10.0 cudnn/10.0-7.5.1.10
srun -u --output=run/l4_w4.outputs sh run/l4_w4.sh