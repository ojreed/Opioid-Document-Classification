# Opioid Classification

This purpose of this repository is to train, test, and implement text classification models for the OIDA dataset. Training documents and trained models are linked here:

[INSERT LINK]

# Usage: 
## Deployment
In order to use our model in order to get a predicted topic for a .txt file or user inputed text please run the main.py file. This code will walk the user through the steps needed to obtain this prediction. In order to change the model used change the file names in the code of `main.py` for the tokenizer and the model imported by pkl.

## Training
In order to train a new model one must run one of the many python files highlighted in the "Training Code" section or referenced in the "File Format Meaning" section of the README. This code will train a model with the following characteristics. To expidite the training process, if the user has access to the ARCH compute cluster, they may submit a job script formatted following the format of the .sh files in the "ARCH script roles" section. The header of these files should remain the same but the user can change the python files that are executed by changing the file names and user inputs at the bottom.


# Doc Info: 

## ARCH Script Roles:

### evaluation_5x.sh
This script executes 5 training runs for each of the SB RNL and Mixed label sets in order to get average model statistics with random train/test splits

### evaluation_data_gen.sh
This script executes 1 training run for each of the SB RNL and Mixed label sets in order to get average model statistics with a fixed seed train/test splits

### OPI_Dual_Label_Survey.sh
This script exectues a set of training runs with different hyper parameters to find the idea setting for training.

### Mod_stats.py
This file generates a real x predicted matrix for a trained model and tokenizer trained on a mixed labelset. The "signle" suffix version does the same for a single label set.

## Training Code

### Mixed_Labeled_BioBERT.py
Specific file for training and testing a `BioBERT` model with 25 epochs and a learning rate of 1e-04 on a the mixed label set. 

## Suport Files

### Opioid_Classification_HF
The `Opioid_Classification_HF` is the main Jupyter Notebook that trains a Hugging Face model on the target dataset. To change the model or tokenizer, simply replace the pretrained layer of `HFClass` and the tokenizer. The training settings can also be adjusted within the notebook.

### Opi Dataset
The `Opi Dataset` is a custom PyTorch dataset used for training. It is easier to store locally, and is used within the `Opioid_Classification_HF` notebook.

### Data Functions
The `Data Functions` directory contains helper functions for prepping the data.

### Pull Text from ID
The `Pull Text from ID` function takes an ID as input and returns the text associated with that ID from an online database.

## File Format Meaning:

### File Format: R suffix
An R at the end of a file name means it is a special version of that file that uses a random rather than seeded train/test split 

### File Format: TOKENIZER_MODEL_EPOCHS_LEARNING RATE.py
Specific file for training and testing a model, tokenizer, hyperparameter combination.

### File Format: SET_NUM_DOCUMENTS_MODEL.py
Specific file for training and testing a BioBERT model with 25 epochs and a learning rate of 1e-04 on a specfic label set with a specific number of documents. 

Please refer to the code and comments within each file for more details on how they are used in the opioid classification process.
