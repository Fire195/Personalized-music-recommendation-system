import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.metrics import accuracy_score
import time

tt = time.time()
threshold = 0.01


def roundToInt(array):
    converted_list = []
    for idx in range(len(array)):
        if array[idx] >= threshold:
            converted_list.append(1)
        else:
            converted_list.append(0)
    return np.array(converted_list)


metrics = ['euclidean', 'manhattan', 'correlation',
           'cosine', 'dice', 'hamming', 'jaccard',
           'sokalmichener', 'sokalsneath', 'sqeuclidean', 'yule',
           'wminkowski', 'nan_euclidean', 'haversine', 'precomputed']
dataset_list = ["dataset500k.csv", "dataset1m.csv",
                "dataset2m.csv", "dataset3m.csv", "dataset4m.csv"]
headers = ['msno', 'song_id', 'genres_id', 'target']

data_name = "dataset500k.csv"
# data_name ="dataset1m.csv"
# data_name ="dataset2m.csv"
# data_name ="dataset3m.csv"
# data_name ="dataset4m.csv"


print(f"current data scale: {str(data_name.replace('.csv',''))[7:]}")

dataset = pd.read_csv(data_name, names=headers)
data = dataset.drop(labels=None, axis=0, index=0,
                    columns=None, inplace=False)
data = data[['msno', 'song_id', 'target']]
# print(data)

users = data.msno.unique()
items = data.song_id.unique()
users_count = data.msno.unique().shape[0]
items_count = data.song_id.unique().shape[0]
print(f"users_count is {users_count}, songs_count is {items_count}")
noneDic = dict()

msno2intDic = noneDic.fromkeys(users)
value = 0
for key in msno2intDic.keys():
    msno2intDic[key] = value
    value += 1

song_id2intDic = noneDic.fromkeys(items)
value = 0
for key in song_id2intDic.keys():
    song_id2intDic[key] = value
    value += 1

train_set, test_set = train_test_split(data, test_size=0.25)
train_data_matrix = np.zeros((users_count, items_count))
test_data_matrix = np.zeros((users_count, items_count))

# print(train_data_matrix.shape)
for line in train_set.itertuples():
    target_value = line[3]
    # print(target_value)
    if target_value != 'target':
        train_data_matrix[msno2intDic[line[1]],
                          song_id2intDic[line[2]]] = line[3]

for line in test_set.itertuples():
    target_value = line[3]
    # print(target_value)
    if target_value != 'target':
        test_data_matrix[msno2intDic[line[1]],
                         song_id2intDic[line[2]]] = line[3]

users_similarity = pairwise_distances(
    train_data_matrix, metric='euclidean')
# items_similarity = pairwise_distances(train_data_matrix.T, metric="cosine")

user_choosing_mean = train_data_matrix.mean(
    axis=1)  # culculate mean in each row
choosing_dif = train_data_matrix - user_choosing_mean[:, np.newaxis]
pred = user_choosing_mean[:, np.newaxis] \
    + users_similarity.dot(choosing_dif) / \
    np.array([np.abs(users_similarity).sum(axis=1)]).T

pred = roundToInt(pred[test_data_matrix.nonzero()].flatten())
test_data_matrix = test_data_matrix[test_data_matrix.nonzero()].flatten()

print("accuracy_score:", accuracy_score(test_data_matrix, pred))
print("using time:", time.time() - tt)
print("cuttent similarity algorithm is euclidean\n")
