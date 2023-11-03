import pickle
import opi_dataset as op
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

"""
USE: File full of useful dataset manipulation functions to reformat the dataset into different formats
"""

def init():
    # init master dataset
    full_dataset = op.OPI_DataSet_Final(data_csv='./opi_2/All_Samples.csv', data_dir='./opi/500_samples',num_labels=1) #init dataset from csv
    print("Init Full Dataset")
    # Creating the dataset and dataloader for the neural network
    # split into train and temp (to split into test and val)
    train_size = int(0.80 * len(full_dataset)) #train is 75% of the dataset
    temp_size = len(full_dataset) - train_size #test becomes the rest
    train_dataset, temp_dataset = torch.utils.data.random_split(full_dataset, [train_size, temp_size]) #do this split randomly
    # split temp dataset into test and val datasets
    test_size = int(0.50 * len(temp_dataset)) #test is 60% of the original test dataset
    val_size = len(temp_dataset) - test_size #test becomes the rest
    test_dataset, val_dataset = torch.utils.data.random_split(temp_dataset, [test_size, val_size]) #do this split randomly

    # save datasets as .pkl files
    with open('./train_dataset.pkl', 'wb') as f:
        pickle.dump(train_dataset, f)
    with open('./test_dataset.pkl', 'wb') as f:
        pickle.dump(test_dataset, f)
    with open('./val_dataset.pkl', 'wb') as f:
        pickle.dump(val_dataset, f)
    with open('./full_dataset.pkl', 'wb') as f:
        pickle.dump(full_dataset, f)
        
def dual_label_csv():
    full_dataset = op.OPI_DataSet_Final(data_csv='./opi_2/All_Samples.csv', data_dir='./opi/500_samples',num_labels=2,returnLabel=2)
    print("Init Full Dataset")
    raw_X = []
    raw_y = []
    raw_z = []
    for item in full_dataset:
        raw_X.append(item[0])
        raw_y.append(item[1])
        raw_z.append(item[2])
    with open('./CSV_raw_dual.pkl', 'wb') as f:
        pickle.dump(pd.DataFrame({'Text': raw_X, 'Raw Label 1': raw_y, 'Raw Label 2': raw_z}), f)
    return

def dual_label_csv_1500():
    full_dataset = op.OPI_DataSet_Final(data_csv='./opi_2/All_1500_Samples.csv', data_dir='./opi/500_samples',num_labels=2,returnLabel=2)
    print("Init Full Dataset")
    raw_X = []
    raw_y = []
    raw_z = []
    for item in full_dataset:
        raw_X.append(item[0])
        raw_y.append(item[1])
        raw_z.append(item[2])
    full_df = pd.DataFrame({'Text': raw_X, 'Raw Label 1': raw_y, 'Raw Label 2': raw_z})
    with open('./CSV_raw_dual_1500.pkl', 'wb') as f:
        pickle.dump(full_df, f)
    print(raw_X)
    return

def load_train_dataset():
    with open('./train_dataset.pkl', 'rb') as f:
        train_dataset = pickle.load(f)
    return train_dataset

def load_test_dataset():
    with open('./test_dataset.pkl', 'rb') as f:
        test_dataset = pickle.load(f)
    return test_dataset

def load_val_dataset():
    with open('./val_dataset.pkl', 'rb') as f:
        val_dataset = pickle.load(f)
    return val_dataset

def load_full_dataset():
    with open('./full_dataset.pkl', 'rb') as f:
        full_dataset = pickle.load(f)
    return full_dataset


def trim_dataset(dataset, percent):
    dataset_size = int(percent * len(dataset))
    return torch.utils.data.random_split(dataset, [dataset_size, len(dataset) - dataset_size])[0]


def init_data_trad():
    full_dataset = op.OPI_DataSet_Final(data_csv='./opi_2/All_Samples.csv', data_dir='./opi/500_samples',num_labels=1) #init dataset from csv
    print("Init Full Dataset")
    raw_X = []
    raw_y = []
    for item in full_dataset:
        raw_X.append(item[0])
        raw_y.append(item[1])

    x_train, x_TEMP, y_train, y_TEMP = train_test_split(raw_X, raw_y,test_size=0.2, random_state=4)
    x_test, x_val, y_test, y_val = train_test_split(raw_X, raw_y,test_size=0.5, random_state=4,)



    # save datasets as .pkl files
    with open('./train_raw.pkl', 'wb') as f:
        pickle.dump([x_train,y_train], f)
    with open('./test_raw.pkl', 'wb') as f:
        pickle.dump([x_test,y_test], f)
    with open('./val_raw.pkl', 'wb') as f:
        pickle.dump([x_val,y_val], f)
    with open('./full_dataset.pkl', 'wb') as f:
        pickle.dump(full_dataset, f)

def init_data_csv():
    full_dataset = op.OPI_DataSet_Final(data_csv='./opi_2/All_Samples.csv', data_dir='./opi/500_samples',num_labels=1) #init dataset from csv
    print("Init Full Dataset")
    raw_X = []
    raw_y = []
    for item in full_dataset:
        raw_X.append(item[0])
        raw_y.append(item[1])
    with open('./CSV_raw.pkl', 'wb') as f:
        pickle.dump(pd.DataFrame({'Text': raw_X, 'Raw Label': raw_y}), f)
    return

def init_data_diff():
    full_dataset = op.OPI_DataSet_Final(data_csv='./opi_2/All_Samples.csv', data_dir='./opi/500_samples',num_labels=1, returnLabel=2) #init dataset from csv
    print("Init Full Dataset")
    raw_X = []
    raw_y = []
    for item in full_dataset:
        if(item[1] != item[2]):
            print(item[1], item[2])
            raw_X.append(item[0])
            raw_y.append(item[1])
    with open('./CSV_diff.pkl', 'wb') as f:
        pickle.dump(pd.DataFrame({'Text': raw_X, 'Raw Label': raw_y}), f)
    return

def init_data_same():
    full_dataset = op.OPI_DataSet_Final(data_csv='./opi_2/All_Samples.csv', data_dir='./opi/500_samples',num_labels=1, returnLabel=2) #init dataset from csv
    print("Init Full Dataset")
    raw_X = []
    raw_y = []
    for item in full_dataset:
        if(item[1] == item[2]):
            print(item[1], item[2])
            raw_X.append(item[0])
            raw_y.append(item[1])
    with open('./CSV_same.pkl', 'wb') as f:
        pickle.dump(pd.DataFrame({'Text': raw_X, 'Raw Label': raw_y}), f)
    return
        
def load_csv():
    with open('./CSV_raw.pkl', 'rb') as f:
        pd = pickle.load(f)
    return pd

def load_train_raw():
    with open('./train_raw.pkl', 'rb') as f:
        train_dataset = pickle.load(f)
    return train_dataset

def load_test_raw():
    with open('./test_raw.pkl', 'rb') as f:
        test_dataset = pickle.load(f)
    return test_dataset

def load_val_raw():
    with open('./val_raw.pkl', 'rb') as f:
        val_dataset = pickle.load(f)
    return val_dataset


# init_data_trad()
# init_data_same()
# dual_label_csv()
# dual_label_csv_1500()
# import pickle
# with open('./CSV_raw_dual_1500.pkl', 'rb') as f:
#     df = pickle.load(f)
# print(df.head)