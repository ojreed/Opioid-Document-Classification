# Importing the libraries needed
import pandas as pd
import torch
import transformers
from transformers import pipeline, RobertaTokenizerFast
from transformers import RobertaTokenizer, RobertaModel
from transformers import AutoTokenizer, AutoModelForMaskedLM
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import Dataset, DataLoader
import data_functions as data
# import wandb

# %%
# Setting up the device for GPU usage


import sys

# Get the integer and float parameters from the command line
EPOCHS = int(sys.argv[1])
LEARNING_RATE = float(sys.argv[2])
RUN_NAME = sys.argv[3]


from torch import cuda
device = 'cuda' if cuda.is_available() else 'cpu'

# %%
# wandb.init(project="OPI Data")

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

# df['ENCODE_CAT_1'] = df['Raw Label 1'].apply(lambda x: encode_cat(x))
# df['ENCODE_CAT_2'] = df['Raw Label 2'].apply(lambda x: encode_cat(x))


df['ENCODE_CAT'] = df['Raw Label 1'].apply(lambda x: encode_cat(x))

# %%
df.head()

# %% [markdown]
# Training Settings

# %%
# Defining some key variables that will be used later on in the training
MAX_LEN = 512
TRAIN_BATCH_SIZE = 25
VALID_BATCH_SIZE = 15


# tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-cased')
# tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base', max_len=512)
tokenizer = AutoTokenizer.from_pretrained('pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb')



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
            'targets': torch.tensor(self.data.ENCODE_CAT[index], dtype=torch.long)
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

# %% [markdown]
# Loss function and Optimizer

# %%
# Creating the loss function and optimizer
loss_function = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params =  model.parameters(), lr=LEARNING_RATE)

# %% [markdown]
# Training functions

# %%
# Function to calcuate the accuracy of the model

def calcuate_accu(big_idx, targets):
    n_correct = (big_idx==targets).sum().item()
    return n_correct

# %%
# Defining the training function on the 80% of the dataset for tuning the distilbert model

def train(epoch):
    tr_loss = 0
    n_correct = 0
    nb_tr_steps = 0
    nb_tr_examples = 0
    model.train()
    for _,data in enumerate(training_loader, 0):
        ids = data['ids'].to(device, dtype = torch.long)
        mask = data['mask'].to(device, dtype = torch.long)
        targets = data['targets'].to(device, dtype = torch.long)

        outputs = model(ids, mask)
        loss = loss_function(outputs, targets)
        tr_loss += loss.item()
        big_val, big_idx = torch.max(outputs.data, dim=1)
        n_correct += calcuate_accu(big_idx, targets)

        nb_tr_steps += 1
        nb_tr_examples+=targets.size(0)
        
        if _%5000==0:
            loss_step = tr_loss/nb_tr_steps
            accu_step = (n_correct*100)/nb_tr_examples 
            print(f"Training Loss per 5000 steps: {loss_step}")
            print(f"Training Accuracy per 5000 steps: {accu_step}")
        
        # if _ % 25 == 0:
        #     wandb.log({"loss": loss})
        
        optimizer.zero_grad()
        loss.backward()
        # # When using GPU
        optimizer.step()

    print(f'The Total Accuracy for Epoch {epoch}: {(n_correct*100)/nb_tr_examples}')
    epoch_loss = tr_loss/nb_tr_steps
    epoch_accu = (n_correct*100)/nb_tr_examples
    print(f"Training Loss Epoch: {epoch_loss}")
    print(f"Training Accuracy Epoch: {epoch_accu}")

    return 

# %%
# wandb.watch(model, log_freq=100)
for epoch in range(EPOCHS):
    train(epoch)

# %% [markdown]
# Validation Functions

# Saving the files for re-use

# output_model_file = './models/LAST_pytorch_distilbert.bin'
# output_vocab_file = './models/LAST_vocab_distilbert.bin'

# model_to_save = model
# torch.save(model_to_save, output_model_file)
# with open(output_vocab_file, 'wb') as f:
#     pickle.dump(tokenizer, f)

# print('All files saved')


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


# %%
acc = valid(model, testing_loader)
print("Accuracy on test data = %0.2f%%" % acc)

# %% [markdown]
# Saving the model

# %%
# Saving the files for re-use
output_model_file = './models/' + str(RUN_NAME) + '_pytorch_distilbert.bin'
output_vocab_file = './models/' + str(RUN_NAME) + '_vocab_distilbert.bin'

model_to_save = model
torch.save(model_to_save, output_model_file)
with open(output_vocab_file, 'wb') as f:
    pickle.dump(tokenizer, f)

print('All files saved')

