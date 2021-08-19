import pandas as pd


file_name = 'test_set.csv'
train_data = pd.read_csv(file_name, usecols=['msno', 'song_id', 'target'])
train_data.to_csv('test_set_CF.csv')
