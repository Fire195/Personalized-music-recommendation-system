# COMP9417
# created in 29th July by Gewei Cheng
# implementation of LightGBM model

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from datetime import datetime
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import lightgbm as lgb


def loadDataSet(train_fileName,test_fileName):
    train = pd.read_csv(train_fileName)
    test = pd.read_csv(test_fileName, names=['msno','song_id','source_system_tab',
                                       'source_screen_name','source_type','target',
                                       'language','city','registered_via','bd',
                                       'gender','song_year','genre_ids'])
    return train, test


def preprocessing(train, test):
    # change the type of features into category and label into unit8
    for col in train.columns:
        if train[col].dtype == object:
            train[col] = train[col].astype('category')
            test[col] = test[col].astype('category')
    train['target'] = train['target'].astype('uint8')
    test['target'] = test['target'].astype('uint8')
    train['language'] = train['language'].astype('category')
    test['language'] = test['language'].astype('category')
    train['city'] = train['city'].astype('category')
    test['city'] = test['city'].astype('category')
    train['registered_via'] = train['registered_via'].astype('category')
    test['registered_via'] = test['registered_via'].astype('category')
    train['bd'] = train['bd'].astype('category')
    test['bd'] = test['bd'].astype('category')
    train['song_year'] = train['song_year'].astype('category')
    test['song_year'] = test['song_year'].astype('category')
    train['genre_ids'] = train['genre_ids'].astype('category')
    test['genre_ids'] = test['genre_ids'].astype('category')
    return train, test

def writeToFile(params):
    learning_rate = str(params['learning_rate'])
    num_leaves = str(params['num_leaves'])
    max_bin = str(params['max_bin'])
    num_iterations = str(params['num_iterations'])
    feature_fraction = str(params['feature_fraction'])
    acc = str(accuracy_score(Y_test, lgbm_preds))
    tm = str(stop - start)
    with open("LGBM_result_records.txt", "a") as f:
        f.write('---------LGBM_result_records---------\n')
        f.write('datasize: 4.0 million\n')
        f.write('learning_rate: ')
        f.write(learning_rate)
        f.write('\n')
        f.write('num_leaves: ')
        f.write(num_leaves)
        f.write('\n')
        f.write('max_bin: ')
        f.write(max_bin)
        f.write('\n')
        f.write('num_iterations: ')
        f.write(num_iterations)
        f.write('\n')
        f.write('feature_fraction: ')
        f.write(feature_fraction)
        f.write('\n')
        f.write('The acu of prediction is: ')
        f.write(acc)
        f.write('\n')
        f.write('learning time: ')
        f.write(tm)
        f.write('\n\n')


train_fileName = 'train_1.csv'
test_fileName = 'test_1.csv'
# train_fileName = 'train_2.csv'
# test_fileName = 'test_2.csv'
# train_fileName = 'train_3.csv'
# test_fileName = 'test_3.csv'
# train_fileName = 'train_4.csv'
# test_fileName = 'test_4.csv'
# train_fileName = 'train_5.csv'
# test_fileName = 'test_5.csv'


# read dataset
train, test = loadDataSet(train_fileName, test_fileName)

# preprocessing
train_processed, test_processed = preprocessing(train, test)


# set params
params = {
            'objective': 'binary',
            'boosting': 'gbdt',
            'learning_rate': 0.1 ,
            'verbose': 0,
            'num_leaves': 2**8,
            'bagging_fraction': 1.0,
            'bagging_freq': 1,
            'bagging_seed': 1,
            'feature_fraction': 0.8,
            'feature_fraction_seed': 1,
            'max_bin': 230,
            'num_iterations':150,
            'metric' : 'auc'
        }


X= train_processed.drop(labels=['target'],axis=1)
Y= train_processed['target'].values
Y= Y.astype('int')
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.1,random_state=60)

tr = lgb.Dataset(X_train, Y_train)
va = lgb.Dataset(X_test, Y_test, reference=tr)

# train model and record time
print('Training LGBM model...')
start = datetime.now()
lgbm_model = lgb.train(params,
                       train_set = tr,
                       valid_sets = va,
                       early_stopping_rounds=10)
stop = datetime.now()

# save model to file
print('Save model...')
lgbm_model.save_model('lightgbm_model.txt')

# predict
print('Start predicting...')
lgbm_preds = lgbm_model.predict(X_test,
                                num_iteration=lgbm_model.best_iteration)

for i in range(len(lgbm_preds)):
    if lgbm_preds[i] >= .5:       # setting threshold to .5
       lgbm_preds[i] = 1
    else:
       lgbm_preds[i] = 0

# output to file
writeToFile(params)

# delete model
del lgbm_model

print('The acu of prediction is:', accuracy_score(Y_test, lgbm_preds))
print('learning time :', stop - start)


