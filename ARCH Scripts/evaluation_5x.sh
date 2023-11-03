#!/bin/bash
#
#SBATCH --job-name=oreed2EvalResults
#SBATCH --output=oreed2EvalResults.out.log
#SBATCH --error=oreed2EvalResults.err.log
#
# Number of tasks needed for this job. Generally, used with MPI jobs
#SBATCH --ntasks=1
#SBATCH --partition=a100
#
# Time format = HH:MM:SS, DD-HH:MM:SS
#SBATCH --time=04:00:00
#
# Number of CPUs allocated to each task.
#SBATCH --cpus-per-task=4
#
# Minimum memory required per allocated  CPU  in  MegaBytes.
#SBATCH --mem-per-cpu=8000
#SBATCH --gres=gpu:2
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
python Mixed_Labeled_BioBERT.py 25 1e-04 Mixed_1500 > Mixed_1500.log

# python Mixed_Labeled_BioBERT_R.py 25 1e-04 Mixed_1500_R1 > Mixed_1500_R1.log
# python Mixed_Labeled_BioBERT_R.py 25 1e-04 Mixed_1500_R2 > Mixed_1500_R2.log
# python Mixed_Labeled_BioBERT_R.py 25 1e-04 Mixed_1500_R3 > Mixed_1500_R3.log
# python Mixed_Labeled_BioBERT_R.py 25 1e-04 Mixed_1500_R4 > Mixed_1500_R4.log
# python Mixed_Labeled_BioBERT_R.py 25 1e-04 Mixed_1500_R5 > Mixed_1500_R5.log


# python s1_1500_BioBERT_R.py 25 1e-04 s1_1500_R1 > s1_1500_R1.log
# python s1_1500_BioBERT_R.py 25 1e-04 s1_1500_R2 > s1_1500_R2.log
# python s1_1500_BioBERT_R.py 25 1e-04 s1_1500_R3 > s1_1500_R3.log
# python s1_1500_BioBERT_R.py 25 1e-04 s1_1500_R4 > s1_1500_R4.log
# python s1_1500_BioBERT_R.py 25 1e-04 s1_1500_R5 > s1_1500_R5.log

# python s2_1500_BioBERT_R.py 25 1e-04 s2_1500_R1 > s2_1500_R1.log
# python s2_1500_BioBERT_R.py 25 1e-04 s2_1500_R2 > s2_1500_R2.log
python s2_1500_BioBERT_R.py 25 1e-04 s2_1500_R3 > s2_1500_R3.log
python s2_1500_BioBERT_R.py 25 1e-04 s2_1500_R4 > s2_1500_R4.log
python s2_1500_BioBERT_R.py 25 1e-04 s2_1500_R5 > s2_1500_R5.log


deactivate
