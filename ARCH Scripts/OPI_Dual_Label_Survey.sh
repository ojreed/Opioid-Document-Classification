#!/bin/bash
#
#SBATCH --job-name=oreed2DualTest
#SBATCH --output=oreed2DualTest.out.log
#SBATCH --error=oreed2DualTest.err.log
#
# Number of tasks needed for this job. Generally, used with MPI jobs
#SBATCH --ntasks=1
#SBATCH --partition=a100
#
# Time format = HH:MM:SS, DD-HH:MM:SS
#SBATCH --time=04:00:00
#
# Number of CPUs allocated to each task.
#SBATCH --cpus-per-task=2
#
# Minimum memory required per allocated  CPU  in  MegaBytes.
#SBATCH --mem-per-cpu=8000
#SBATCH --gres=gpu:1
#SBATCH -A angieliu_gpu
#
# Send mail to the email address when the job fails
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=oreed2@jhu.edu



# Load necessary modules
ml purge
ml gcc/9.3.0
ml cuda/11.1.0
ml anaconda/2020.07
ml pyTorch/1.8.1-cuda-11.1.1


# Set the number of GPUs per node and export CUDA_VISIBLE_DEVICES
GPUS_PER_NODE=1
export CUDA_VISIBLE_DEVICES=0

# Change to directory where the python script is located
cd /scratch4/angieliu/Opi_Files/

#setup python
module load python/3.8.6
python3 -m venv myenv
source myenv/bin/activate
pip install pandas torch transformers
pip install torchvision
pip install scikit-learn

# Run the Python script
python Mixed_Labeled_BioBERT.py 15 1e-04 ShortFast_newLoss > Mixed_Labeled_BioBERT_newLoss1.log
python Mixed_Labeled_BioBERT.py 25 1e-04 MidFast_newLoss > Mixed_Labeled_BioBERT_newLoss2.log
python Mixed_Labeled_BioBERT.py 25 1e-05 MidSlow_newLoss > Mixed_Labeled_BioBERT_newLoss3.log
python Mixed_Labeled_BioBERT.py 35 1e-04 LongFast_newLoss > Mixed_Labeled_BioBERT_newLoss4.log
deactivate
