# -*- coding: utf-8 -*-
"""
Created on Tue Aug  4 15:25:05 2020

@author: ASUS
"""

import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from datetime import datetime
df = pd.read_csv("train_5.csv")#the name of file
df=df.drop(['msno'],axis=1)
df=df.drop(['song_id'],axis=1)
#df=df.drop(["Unnamed: 0"],axis=1) 
df= df.fillna(0)
df.gender [df.gender == 'female'] =1
df.gender [df.gender == 'male'] =0
for col in df.columns:
    if df[col].dtype == object:
        df[col] = df[col].astype('category')


#print(df.info())
source_tab = list()#types in source_tab
count_tab =0#the number of count tab
for a in df.source_system_tab:
    if a not in source_tab:#Not in the list to indicate that it has not been replaced
        df=df.replace(a,count_tab)
        source_tab.append(a)
        #print(count_tab)
        count_tab = count_tab+1
#print(df.source_system_tab)
        
  #  else:
   #     df.source_system_tab[df.source_system_tab] = source_tab[a]
screen_name = list()
count_screen = 0
for b in df.source_screen_name:
    if b not in screen_name:
        df=df.replace(b,count_screen)
        screen_name.append(b)
        count_screen = count_screen+1
 #   else:
  #      df.source_screen_name[df.source_screen_name] = count_screen[b]
type_list = list()
count_type = 0
for c in df.source_type:
    if c not in type_list:
        df=df.replace(c,count_type)
        type_list.append(c)
        count_type = count_type+1

#print("end")
#print(df.head())
Y= df['target'].values
Y= Y.astype('int')

X=df.drop(labels=['target'],axis=1)

from sklearn.model_selection import train_test_split
#0.9 ;0.7;0.5;0.3;0.1
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.1,random_state=60)
#print(X_train)
start = datetime.now() 
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=250,random_state=None,bootstrap=True,class_weight=None,criterion='gini',max_depth=25,
                               max_features='auto',max_leaf_nodes=None,min_impurity_decrease=0.0,min_impurity_split=None,min_samples_leaf=1,min_samples_split=2,min_weight_fraction_leaf=0.0,
                               n_jobs=1,oob_score=False,verbose=0,warm_start=False)
#print("class end")
model.fit(X_train,Y_train)
prediction_test= model.predict(X_test)
from sklearn import metrics
accuracy = metrics.accuracy_score(Y_test,prediction_test)
stop = datetime.now()
#print(prediction_test)#prediction
print("time:",stop - start)
print('Accuracy=',metrics.accuracy_score(Y_test,prediction_test))

