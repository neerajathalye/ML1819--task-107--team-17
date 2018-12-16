# Import libraries

import pandas as pd
import re
import os.path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

#from sklearn_pandas import DataFrameMapper # Notice that this is actually Sklearn-Pandas library
#%matplotlib inline

# Load dataset
dataset_path = os.path.join(os.path.dirname(os.path.dirname(os.getcwd())), 'dataset\\gender-classifier-DFE-791531.csv')
data = pd.read_csv(dataset_path, encoding='latin1')
#data = pd.read_csv('../input/gender-classifier-DFE-791531.csv', encoding='latin-1')

# Drop unnecessary columns/features
data.drop (columns = ['_unit_id',
                      '_last_judgment_at',
                      'user_timezone',
                      'tweet_coord',
                      'tweet_count',
                      'tweet_created',
                      'tweet_id',
                      'tweet_location',
                      'profileimage',
                      'created'], inplace = True)

data.info()

print(data.head(3))

#Cleaning Dataset
#Analysing the Gender column
print(data['gender'].value_counts())

#Removing the unknown parameters in the Gender
drop_items_idx1 = data[data['gender'] == 'unknown'].index
drop_items_idx2 = data[data['gender'] == 'brand'].index

data.drop (index = drop_items_idx1, inplace = True)
data.drop (index = drop_items_idx2, inplace = True)

print(data['gender'].value_counts())

#Handling the missing dataset values by deleting the NaN values

print ('profile_yn information:\n',data['profile_yn'].value_counts())

print(data[data['profile_yn'] == 'no']['gender'])
print(data[data['profile_yn'] == 'yes']['gender'])

drop_items_idx3 = data[data['profile_yn'] == 'no'].index
drop_items_idx4 = data[data['profile_yn'] == 'yes'].index

data.drop (index = drop_items_idx3, inplace = True)
data.drop (index = drop_items_idx4, inplace = True)

data['profile_yn'].value_counts()
data.drop (columns = ['profile_yn','profile_yn:confidence','profile_yn_gold'], inplace = True)

#Checking the data again after some modifications(cleaning/handling missing values)

print (data['gender'].value_counts())

print ('++++++++++++++++++++++++++++')
data.info()


