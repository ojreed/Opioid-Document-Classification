import opi_dataset as op
import pandas as pd
import pickle

"""
USE: Find label set discrepency breakdown and generate 5x5 relation chart
"""

mode = 0

def encode_cat(x):
	if x not in encode_dict.keys():
		encode_dict[x]=len(encode_dict)
	return encode_dict[x]


if mode == 1:
	full_dataset = op.OPI_DataSet_Dual(data_csv='./opi_2/All_Samples.csv', data_dir='./opi/500_samples',num_labels=1) #init dataset from csv
	print("Init Full Dataset")
	raw_X = []
	raw_y_1 = []
	raw_y_2 = []

	for item in full_dataset:
		raw_X.append(item[0])
		raw_y_1.append(item[1][0])
		raw_y_2.append(item[1][1])
	df = pd.DataFrame({'Text': raw_X, 'Raw Label 1': raw_y_1, 'Raw Label 2': raw_y_2})
	encode_dict = {}

	df['ENCODE_CAT'] = df['Raw Label 1'].apply(lambda x: encode_cat(x))

	with open('./CSV_raw_stats.pkl', 'wb') as f:
		pickle.dump(pd.DataFrame(df), f)
if mode == 0:
	with open('./CSV_raw_stats.pkl', 'rb') as f:
	    df = pickle.load(f)



print(df.head())

train_size = 0.8
train_dataset=df.sample(frac=train_size,random_state=200)
test_dataset=df.drop(train_dataset.index).reset_index(drop=True)
train_dataset = train_dataset.reset_index(drop=True)

train_stats_1 = {}
test_stats_1 = {}
full_stats_1 = {}
train_stats_2 = {}
test_stats_2 = {}
full_stats_2 = {}
train_agree = []
test_agree = []
full_agree = []


one_to_two = {}


for index, row in train_dataset.iterrows():
	if row["Raw Label 1"] not in one_to_two.keys():
		one_to_two[row["Raw Label 1"]] = {}
	if row["Raw Label 2"] not in one_to_two[row["Raw Label 1"]].keys():
		one_to_two[row["Raw Label 1"]][row["Raw Label 2"]] = 1
	else:
		one_to_two[row["Raw Label 1"]][row["Raw Label 2"]] += 1
	if row["Raw Label 1"] not in full_stats_1.keys():
		full_stats_1[row["Raw Label 1"]] = 1
	else:
		full_stats_1[row["Raw Label 1"]] += 1
	if row["Raw Label 2"] not in full_stats_2.keys():
		full_stats_2[row["Raw Label 2"]] = 1
	else:
		full_stats_2[row["Raw Label 2"]] += 1
	if row["Raw Label 1"] in train_stats_1.keys():
		train_stats_1[row["Raw Label 1"]] += 1
		
	else:
		train_stats_1[row["Raw Label 1"]] = 1
	if row["Raw Label 2"] in train_stats_2.keys():
		train_stats_2[row["Raw Label 2"]] += 1	
	else:
		train_stats_2[row["Raw Label 2"]] = 1
	train_agree.append(row["Raw Label 1"] == row["Raw Label 2"])
	full_agree.append(row["Raw Label 1"] == row["Raw Label 2"])

for index, row in test_dataset.iterrows():
	if row["Raw Label 1"] not in one_to_two.keys():
		one_to_two[row["Raw Label 1"]] = {}
	if row["Raw Label 2"] not in one_to_two[row["Raw Label 1"]].keys():
		one_to_two[row["Raw Label 1"]][row["Raw Label 2"]] = 1
	else:
		one_to_two[row["Raw Label 1"]][row["Raw Label 2"]] += 1
	if row["Raw Label 1"] not in full_stats_1.keys():
		full_stats_1[row["Raw Label 1"]] = 1
	else:
		full_stats_1[row["Raw Label 1"]] += 1
	if row["Raw Label 2"] not in full_stats_2.keys():
		full_stats_2[row["Raw Label 2"]] = 1
	else:
		full_stats_2[row["Raw Label 2"]] += 1
	if row["Raw Label 1"] in test_stats_1.keys():
		test_stats_1[row["Raw Label 1"]] += 1
	else:
		test_stats_1[row["Raw Label 1"]] = 1
	if row["Raw Label 2"] in test_stats_2.keys():
		test_stats_2[row["Raw Label 2"]] += 1
	else:
		test_stats_2[row["Raw Label 2"]] = 1
	
	test_agree.append(row["Raw Label 1"] == row["Raw Label 2"])
	full_agree.append(row["Raw Label 1"] == row["Raw Label 2"])



print("train 1",train_stats_1)
print("train 2",train_stats_2)
print("test 1",test_stats_1)
print("test 2",test_stats_2)
print("full 1",full_stats_1)
print("full 2",full_stats_2)
print("average agreement full", (sum(full_agree)/len(full_agree)))
print(one_to_two)