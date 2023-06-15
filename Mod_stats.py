import opi_dataset as op
import pandas as pd
import pickle
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertModel, DistilBertTokenizer
from transformers import pipeline, RobertaTokenizerFast
from transformers import RobertaTokenizer, RobertaModel
from transformers import AutoTokenizer, AutoModelForMaskedLM
from transformers import AutoTokenizer, AutoModel
import data_functions as data
import numpy as np
import pandas as pd
import torch
import transformers
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForMaskedLM
from transformers import AutoTokenizer, AutoModel
import data_functions as data



"""
USE: Generates 5x5 real_pred matrix for a model
"""


# %%
# Setting up the device for GPU usage

from torch import cuda
device = 'cuda' if cuda.is_available() else 'cpu'

# %%
# wandb.init(project="OPI Data")

EPOCHS = 15
LEARNING_RATE = 1e-04
RUN_NAME = "Test"

# %%
import pickle
with open('./CSV_raw_dual.pkl', 'rb') as f:
    df = pickle.load(f)
# # Converting the codes to appropriate categories using a dictionary



encode_dict = {}

def encode_cat(x):
    if x not in encode_dict.keys():
        encode_dict[x]=len(encode_dict)
    return encode_dict[x]

df['ENCODE_CAT_1'] = df['Raw Label 1'].apply(lambda x: encode_cat(x))
df['ENCODE_CAT_2'] = df['Raw Label 2'].apply(lambda x: encode_cat(x))

print(encode_dict)

# %% [markdown]
# Training Settings

# %%
# Defining some key variables that will be used later on in the training
MAX_LEN = 512
TRAIN_BATCH_SIZE = 25
VALID_BATCH_SIZE = 15


# Load the tokenizer using pickle
with open('./Results/Mixed_1000_vocab_distilbert.bin', 'rb') as handle:
    tokenizer = pickle.load(handle)

# tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-cased')
# tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base', max_len=512)
# tokenizer = AutoTokenizer.from_pretrained('pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb')



# %%
class Triage(Dataset):
    def __init__(self, dataframe, tokenizer, max_len):
        self.len = len(dataframe)
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_len = max_len
        
    def __getitem__(self, index):
        title = str(self.data.Text[index])
        title = " ".join(title.split())
        inputs = self.tokenizer.encode_plus(
            title,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            pad_to_max_length=True,
            return_token_type_ids=True,
            truncation=True
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']

        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'targets': torch.tensor([self.data.ENCODE_CAT_1[index],self.data.ENCODE_CAT_2[index]], dtype=torch.long)
        } 
    
    def __len__(self):
        return self.len

# %%
# Creating the dataset and dataloader for the neural network

train_size = 0.8
train_dataset=df.sample(frac=train_size,random_state=200)
test_dataset=df.drop(train_dataset.index).reset_index(drop=True)
train_dataset = train_dataset.reset_index(drop=True)


print("FULL Dataset: {}".format(df.shape))
print("TRAIN Dataset: {}".format(train_dataset.shape))
print("TEST Dataset: {}".format(test_dataset.shape))

training_set = Triage(train_dataset, tokenizer, MAX_LEN)
testing_set = Triage(test_dataset, tokenizer, MAX_LEN)

# %%
train_params = {'batch_size': TRAIN_BATCH_SIZE,
                'shuffle': True,
                'num_workers': 0
                }

test_params = {'batch_size': VALID_BATCH_SIZE,
                'shuffle': True,
                'num_workers': 0
                }

training_loader = DataLoader(training_set, **train_params)
testing_loader = DataLoader(testing_set, **test_params)

# %%
# Creating the customized model, by adding a drop out and a dense layer on top of distil bert to get the final output for the model. 

class HFClass(torch.nn.Module):
    def __init__(self):
        super(HFClass, self).__init__()
        # self.l1 = DistilBertModel.from_pretrained("distilbert-base-uncased")
        # self.l1 = RobertaModel.from_pretrained('roberta-base')
        self.l1 = AutoModel.from_pretrained('pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb')
        self.pre_classifier = torch.nn.Linear(768, 768)
        self.dropout = torch.nn.Dropout(0.3)
        self.classifier = torch.nn.Linear(768, len(encode_dict))

    def forward(self, input_ids, attention_mask):
        output_1 = self.l1(input_ids=input_ids, attention_mask=attention_mask)
        hidden_state = output_1[0]
        pooler = hidden_state[:, 0]
        pooler = self.pre_classifier(pooler)
        pooler = torch.nn.ReLU()(pooler)
        pooler = self.dropout(pooler)
        output = self.classifier(pooler)
        return output

# %%
model = HFClass()
model.to(device)


# Load the PyTorch model from the .bin file
model = torch.load('./Results/Mixed_1000_pytorch_distilbert.bin',map_location=torch.device('cpu'))


# %% [markdown]
# Loss function and Optimizer

import torch.nn as nn


class CustomLoss(nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()

    def forward(self, outputs, labels):
        # Convert outputs to predicted labels using argmax
        predicted_labels = torch.argmax(outputs, dim=1)

        # Check which labels match the predicted labels
        matches = (labels[:,0] == predicted_labels) | (labels[:,0] == predicted_labels)

        # Compute the loss for each label
        loss_1 = nn.CrossEntropyLoss()(outputs, labels[:, 0])
        loss_2 = nn.CrossEntropyLoss()(outputs, labels[:, 1])

        # Compute the minimum loss between the two labels
        loss = torch.min(loss_1, loss_2)

        # Mask the losses for incorrect predictions
        masked_loss = torch.where(matches, loss, torch.tensor(0.0, device=outputs.device))

        # Average the masked losses across the batch
        loss = torch.mean(masked_loss)

        return loss




# %%
# Creating the loss function and optimizer
# loss_function = torch.nn.CrossEntropyLoss()
loss_function = CustomLoss()
optimizer = torch.optim.Adam(params =  model.parameters(), lr=LEARNING_RATE)



# %% [markdown]
# Training functions

# %%
# Function to calcuate the accuracy of the model

def calcuate_accu(big_idx, targets):
    n_correct = 0
    for index in range(0, len(big_idx)):
        n_correct += max(big_idx[index] == targets[index,0],big_idx[index] == targets[index,1])
    return n_correct






real_pred = [[],[]]

# %%
def valid(model, testing_loader):
    tr_loss = 0
    n_correct = 0
    nb_tr_steps = 0
    nb_tr_examples = 0
    model.eval()
    n_correct = 0; n_wrong = 0; total = 0
    with torch.no_grad():
        for _, data in enumerate(testing_loader, 0):
            ids = data['ids'].to(device, dtype = torch.long)
            mask = data['mask'].to(device, dtype = torch.long)
            targets = data['targets'].to(device, dtype = torch.long)
            outputs = model(ids, mask).squeeze()
            loss = loss_function(outputs, targets)
            tr_loss += loss.item()
            big_val, big_idx = torch.max(outputs.data, dim=1)
            # print(big_idx)
            # print(targets)
            real_pred[1] += big_idx.tolist()
            real_pred[0] += targets.tolist()
            # print(real_pred)
            n_correct += calcuate_accu(big_idx, targets)
            nb_tr_steps += 1
            nb_tr_examples+=targets.size(0)
            
            if _%5000==0:
                loss_step = tr_loss/nb_tr_steps
                accu_step = (n_correct*100)/nb_tr_examples
                print(f"Validation Loss per 100 steps: {loss_step}")
                print(f"Validation Accuracy per 100 steps: {accu_step}")
    epoch_loss = tr_loss/nb_tr_steps
    epoch_accu = (n_correct*100)/nb_tr_examples
    print(f"Validation Loss Epoch: {epoch_loss}")
    print(f"Validation Accuracy Epoch: {epoch_accu}")
    
    return epoch_accu




print(real_pred)

# Set the model to evaluation mode
model.eval()

# %%
acc = valid(model, testing_loader)
print("Accuracy on test data = %0.2f%%" % acc)

#SET ONE
real_to_pred = np.zeros((5,5))
for idx in range(len(real_pred[0])):
	real_to_pred[real_pred[0][idx][0]][real_pred[1][idx]] +=1

print(real_to_pred)

# Calculate the norm of each row
norms = np.linalg.norm(real_to_pred, axis=1)

# Divide each row by its norm using broadcasting
normalized_matrix = real_to_pred / norms[:, np.newaxis]

print(normalized_matrix)

#SET TWO
real_to_pred = np.zeros((5,5))
for idx in range(len(real_pred[0])):
    real_to_pred[real_pred[0][idx][1]][real_pred[1][idx]] +=1

print(real_to_pred)

# Calculate the norm of each row
norms = np.linalg.norm(real_to_pred, axis=1)

# Divide each row by its norm using broadcasting
normalized_matrix = real_to_pred / norms[:, np.newaxis]

print(normalized_matrix)