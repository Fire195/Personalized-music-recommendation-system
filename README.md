##########################################################
following lines are the readme file for "cf1.py"
python version: 3.7
sklearn version 0.21.1
please ensure running machine with current version

we use extra packages listed below:
import numpy as np
import pandas as pd
import time
import lightgbm as lgb
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import pairwise_distances, mean_squared_error
from math import sqrt
from sklearn.metrics import accuracy_score

 please ensure your environment installed those packages.

please ensure your model files(.py) and all dataset(.cdv) files all in the same directory
if using terminal or other consoles, 
run with command `python3 cf.py > output_CF.txt`,
                 `python3 random_forest.py > output_RF.txt` or 
                 `python3 LigntGBM.py`
and you may need to wait for a long time.
it will automatically create an `.txt` file,
there should be:
    current testing dataset scale,
    type of current using similarity algorithm.
    modle accuracy of corresponding configues.
   
potential issues :
1. if you got error like this:
    ImportError: No module named lightgbm
    
you can use linux command below to solve this issue: 
    pip install lightgbm
    
2. if you got error like this:
    OSError: dlopen(/Users/{xxx}/anaconda3/lib/python3.6/site-packages/lightgbm/lib_lightgbm.so, 6): Library not loaded: /usr/local/opt/gcc/lib/gcc/7/libgomp.1.dylib
Referenced from: /Users/{xxx}/anaconda3/lib/python3.6/site-packages/lightgbm/lib_lightgbm.so
Reason: image not found

you can use linux command below to solve this issue:
    brew install libomp 
    
Above lines are the readme file for this Project 
###########################################################