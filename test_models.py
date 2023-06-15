import torch
from transformers import BertTokenizer


"""
USE:Generates F1 score stats for a model 
"""


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

def encode_cat(x):
    if x not in encode_dict.keys():
        encode_dict[x]=len(encode_dict)
    return encode_dict[x]


def load_model(model_path):
    """
    Load PyTorch model from the given path.
    """
    model = torch.load(model_path, map_location=torch.device('cpu'))
    return model

def load_vocab(vocab_path):
    """
    Load vocabulary from the given path using the BERT tokenizer.
    """
    tokenizer = BertTokenizer.from_pretrained(vocab_path)
    return tokenizer

def calculate_f1_score(model, dataloader):
    """
    Calculate the F1 score of the given PyTorch model on the given dataset.
    """
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in dataloader:
            inputs, labels = data
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    precision = correct / total
    recall = correct / total
    f1_score = 2 * (precision * recall) / (precision + recall)
    return f1_score

def calculate_label_percentage(model, dataloader, label_count):
    """
    Calculate the percentage breakdown of each label predicted by the given PyTorch model on the given dataset.
    """
    model.eval()
    label_counts = [0] * label_count
    total = 0
    with torch.no_grad():
        for data in dataloader:
            inputs, labels = data
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            for i in range(len(predicted)):
                label_counts[predicted[i]] += 1
            total += labels.size(0)
    label_percentages = [count / total for count in label_counts]
    return label_percentages

if __name__ == '__main__':
    # Load model and vocabulary
    model_path = 'path/to/model.pt'
    vocab_path = 'path/to/vocab'
    model = load_model(model_path)
    tokenizer = load_vocab(vocab_path)

    # Load dataset and create dataloader
    # %%
    import pickle
    with open('./CSV_raw_dual.pkl', 'rb') as f:
        df = pickle.load(f)
    # # Converting the codes to appropriate categories using a dictionary



    encode_dict = {}

    df['ENCODE_CAT_1'] = df['Raw Label 1'].apply(lambda x: encode_cat(x))
    df['ENCODE_CAT_2'] = df['Raw Label 2'].apply(lambda x: encode_cat(x))

    train_size = 0.8
    train_dataset=df.sample(frac=train_size,random_state=200)
    test_dataset=df.drop(train_dataset.index).reset_index(drop=True)
    train_dataset = train_dataset.reset_index(drop=True)


    print("FULL Dataset: {}".format(df.shape))
    print("TRAIN Dataset: {}".format(train_dataset.shape))
    print("TEST Dataset: {}".format(test_dataset.shape))

    training_set = Triage(train_dataset, tokenizer, MAX_LEN)
    testing_set = Triage(test_dataset, tokenizer, MAX_LEN)

    dataloader = DataLoader(testing_set, batch_size=32)

    # Calculate F1 score
    f1_score = calculate_f1_score(model, dataloader)
    print(f'F1 score: {f1_score:.4f}')

    # Calculate label percentage breakdown
    label_count = 5
    label_percentages = calculate_label_percentage(model, dataloader, label_count)
    for i in range(len(label_percentages)):
        print(f'Label {i+1}: {label_percentages[i]*100:.2f}%')
