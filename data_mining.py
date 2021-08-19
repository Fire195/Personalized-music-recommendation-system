# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
# import lightgbm as lgb

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "./kk"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

print('Loading data...')
data_path = "./kk"
train = pd.read_csv(data_path + '/train.csv', dtype={'msno' : 'category',
                                                'source_system_tab' : 'category',
                                                  'source_screen_name' : 'category',
                                                  'source_type' : 'category',
                                                  'target' : np.uint8,
                                                  'song_id' : 'category'})
for i in range(100):
    print(train[i])
# test = pd.read_csv(data_path + 'test.csv', dtype={'msno' : 'category',
#                                                 'source_system_tab' : 'category',
#                                                 'source_screen_name' : 'category',
#                                                 'source_type' : 'category',
#                                                 'song_id' : 'category'})
# songs = pd.read_csv(data_path + 'songs.csv',dtype={'genre_ids': 'category',
#                                                   'language' : 'category',
#                                                   'artist_name' : 'category',
#                                                   'composer' : 'category',
#                                                   'lyricist' : 'category',
#                                                   'song_id' : 'category'})
# members = pd.read_csv(data_path + 'members.csv',
#                       dtype={'city' : 'category',
#                              'bd' : np.uint8,
#                              'gender' : 'category',
#                              'registered_via' : 'category'}
#                      ,parse_dates=["registration_init_time","expiration_date"])
# songs_extra = pd.read_csv(data_path + 'song_extra_info.csv')

print('Data preprocessing...')