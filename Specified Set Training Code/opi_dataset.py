import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torchvision import transforms, utils
import pull_text_from_id as site
import os


"""
USE: datastructures representing the 500 and 1000 size datasets in the pytorch dataset format 
"""


class OPI_DataSet_500(Dataset):
	def __init__(self, data_csv, data_dir, transform=None,num_labels=1):
		self.label_csv = pd.read_csv(data_csv) #FOR US: "./opi/Liu_500_sample.csv")
		self.data_dir = data_dir #FOR US: ./opi/500_samples
		self.transform = transform
		self.num_labels = num_labels

	def __len__(self):
		return len(self.label_csv) #returns the number of elements in the dataset --> should result in 500

	def __getitem__(self, idx):
		if torch.is_tensor(idx):
			idx = idx.tolist()
		#get text
		doc_id = self.label_csv.iloc[idx, 1]
		docText = ""
		if os.path.exists(str(self.data_dir)+"/"+str(doc_id)+"/"+str(doc_id)+".ocr"):
			f = open(str(self.data_dir)+"/"+str(doc_id)+"/"+str(doc_id)+".ocr", "r", encoding="utf8")
			f.seek(0)
			for line in f: # goes through all lines
				for word in line.lower().split(): # goes throuugh all words per line
					# pureWord = word.strip("[]}{,.\\/!@#$%^&*()<>#;?''" '"')
					pureWord = word
					docText += pureWord
					docText += " "
			f.close()
		else:	
			rawText = site.pull_text(str(doc_id))
			f = open(rawText, "r", encoding="utf8",errors='ignore')
			f.seek(0)
			for line in f: # goes through all lines
				# print(line)
				for word in line.lower().split(): # goes throuugh all words per line
					# pureWord = word.strip("[]}{,.\\/!@#$%^&*()<>#;?''" '"')
					pureWord = word
					docText += pureWord
					docText += " "
			f.close()

		#get labels
		doc_labels = [self.label_csv.iloc[idx, doc_label] for doc_label in range(2,2+self.num_labels)] 
		sample = {'text': docText, 'labels': doc_labels}

		if self.transform:
			sample = self.transform(sample)

		return (sample['text'], sample['labels'][0])


class OPI_DataSet_Final(Dataset):
	def __init__(self, data_csv, data_dir, transform=None,num_labels=1, returnLabel=1):
		self.label_csv = pd.read_csv(data_csv) #FOR US: "./opi/Liu_500_sample.csv")
		self.data_dir = data_dir #FOR US: ./opi/500_samples
		self.transform = transform
		self.num_labels = num_labels
		self.encode_dict = {}
		self.cached = {}
		self.returnLabel = returnLabel

	def __len__(self):
		return len(self.label_csv) #returns the number of elements in the dataset --> should result in 500

	def __getitem__(self, idx):
		if idx in self.cached.keys():
			return self.cached[idx]
		if torch.is_tensor(idx):
			idx = idx.tolist()
		#get text
		doc_id = self.label_csv.iloc[idx, 0]
		docText = ""
		if os.path.exists(str(self.data_dir)+"/"+str(doc_id)+"/"+str(doc_id)+".ocr"):
			f = open(str(self.data_dir)+"/"+str(doc_id)+"/"+str(doc_id)+".ocr", "r", encoding="utf8")
			f.seek(0)
			for line in f: # goes through all lines
				for word in line.lower().split(): # goes throuugh all words per line
					# pureWord = word.strip("[]}{,.\\/!@#$%^&*()<>#;?''" '"')
					pureWord = word
					docText += pureWord
					docText += " "
			f.close()
		else:	
			# print("Miss")
			rawText = site.pull_text(str(doc_id))
			f = open(rawText, "r", encoding="utf8",errors='ignore')
			f.seek(0)
			for line in f: # goes through all lines
				# print(line)
				for word in line.lower().split(): # goes throuugh all words per line
					# pureWord = word.strip("[]}{,.\\/!@#$%^&*()<>#;?''" '"')
					pureWord = word
					docText += pureWord
					docText += " "
			f.close()

		#get labels
		doc_labels = [self.label_csv.iloc[idx, 1],self.label_csv.iloc[idx, 2]] 
		sample = {'text': docText, 'labels': doc_labels}

		if self.transform:
			sample = self.transform(sample)
		self.cached[idx] = (sample['text'], sample['labels'][0])
		if(self.returnLabel == 1):
			return (sample['text'], sample['labels'][0])
		else:
			return (sample['text'], sample['labels'][0], sample['labels'][1])




#test code
# opi_dataset = OPI_DataSet(data_csv='./opi/Liu_500_sample.csv',
# 									data_dir='./opi/500_samples',num_labels=1)

# for i in range(len(opi_dataset)):
# 	print(opi_dataset[i])


