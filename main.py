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
from torch import cuda
device = 'cuda' if cuda.is_available() else 'cpu'


with open('./models/Mixed_1000_vocab_distilbert.bin', 'rb') as handle:
    tokenizer = pickle.load(handle)

"""
This main file serves as an example of how to use this model to classify text. The text will prompt the user to either input raw text or a file name, 
    then the model will return a prediction. 

For users hoping to implment in an alternative way with one of our trained models keep all code before line 83 
    and modify the use case of "prediction = get_prediction(text)"
"""



class HFClass(torch.nn.Module):
    def __init__(self):
        super(HFClass, self).__init__()
        self.l1 = AutoModel.from_pretrained('pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb')
        self.pre_classifier = torch.nn.Linear(768, 768)
        self.dropout = torch.nn.Dropout(0.3)
        self.classifier = torch.nn.Linear(768, 5)

    def forward(self, input_ids, attention_mask):
        output_1 = self.l1(input_ids=input_ids, attention_mask=attention_mask)
        hidden_state = output_1[0]
        pooler = hidden_state[:, 0]
        pooler = self.pre_classifier(pooler)
        pooler = torch.nn.ReLU()(pooler)
        pooler = self.dropout(pooler)
        output = self.classifier(pooler)
        return output


# Load the PyTorch model from the .bin file
model = HFClass()
model = torch.load('./models/Mixed_1000_pytorch_distilbert.bin', map_location=torch.device('cpu'))
model.to(device)

def get_prediction(text):
    # Tokenize the input text
    inputs = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        truncation=True,
        padding='max_length',
        max_length=512,
        return_tensors='pt'
    )

    # Get the input tensors
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)

    # Forward pass through the model
    with torch.no_grad():
        outputs = model(input_ids, attention_mask)

    # Get the predicted label
    predicted_label = torch.argmax(outputs, dim=1).item()

    return predicted_label


# Prompt the user for input
choice = input("Enter '1' to input the name of a .txt file or '2' to write the text: ")

if choice == '1':
    file_path = input("Enter the path to the .txt file: ")

    with open(file_path, 'r') as file:
        text = file.read()

    prediction = get_prediction(text)
    print("Predicted label:", prediction)

elif choice == '2':
    text = input("Enter the text: ")
    prediction = get_prediction(text)
    print("Predicted label:", prediction)

else:
    print("Invalid choice.")
