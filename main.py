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


with open('./Results/Mixed_1000_vocab_distilbert.bin', 'rb') as handle:
    tokenizer = pickle.load(handle)


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


# Load the PyTorch model from the .bin file
model = HFClass()
model.to(device)
model.load_state_dict(torch.load('./Results/Mixed_1000_pytorch_distilbert.bin', map_location=torch.device('cpu')))


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
