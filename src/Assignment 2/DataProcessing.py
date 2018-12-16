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

# data.info()

# print(data.head(3))

#Cleaning Dataset
#Analysing the Gender column
# print(data['gender'].value_counts())

#Removing the unknown parameters in the Gender
drop_items_idx1 = data[data['gender'] == 'unknown'].index
drop_items_idx2 = data[data['gender'] == 'brand'].index

data.drop (index = drop_items_idx1, inplace = True)
data.drop (index = drop_items_idx2, inplace = True)

# print(data['gender'].value_counts())

#Handling the missing dataset values by deleting the NaN values(all rows having profile_yn = 'no'

# print ('profile_yn information:\n',data['profile_yn'].value_counts())

# print(data[data['profile_yn'] == 'no']['gender'])


drop_items_idx3 = data[data['profile_yn'] == 'no'].index

data.drop (index = drop_items_idx3, inplace = True)

# print ('profile_yn information:\n',data['profile_yn'].value_counts())
data.drop (columns = ['profile_yn','profile_yn:confidence','profile_yn_gold'], inplace = True)

#Checking the data again after some modifications(cleaning/handling missing values)

# print (data['gender'].value_counts())

# print ('++++++++++++++++++++++++++++')
# data.info()

#Checking the Gender Confidence column, selecting only the values with gender confidence = 1 and dropping others

# print ('Total data: ', data.shape)
# print ('Data with gender confidence < 1: ', data[data['gender:confidence'] < 1].shape)

drop_items_idx4 = data[data['gender:confidence'] < 1].index
data.drop (index = drop_items_idx4, inplace = True)
# print (data['gender:confidence'].value_counts())
data.drop (columns = ['gender:confidence'], inplace = True)

#Again deleting some less useful features as a part of feature selection

data.drop (columns = ['_golden','_unit_state','_trusted_judgments','gender_gold'], inplace = True)

# Checking the data
# print (data['gender'].value_counts())

# print ('++++++++++++++++++++++++++')
# data.info()

#Now, for differenciate among features, visualizing the columns

#Visualizing gender using countplot
# sns.countplot(data['gender'],label="Gender")
# plt.show()
#
#
# # #Visualizing the amount of tweet favourites and retweets
# sns.barplot (x = 'gender', y = 'fav_number',data = data)
# plt.show()
# sns.barplot (x = 'gender', y = 'retweet_count',data = data)
# plt.show()
#
#Visualizing colour attributes - sidebar colour (Male)
#
# male_top_sidebar_color = data[data['gender'] == 'male']['sidebar_color'].value_counts()
# print(male_top_sidebar_color)
# male_top_sidebar_color = data[data['gender'] == 'male']['sidebar_color'].value_counts().head(6)
# print(male_top_sidebar_color)
# male_top_sidebar_color_idx = male_top_sidebar_color.index
# male_top_color = male_top_sidebar_color_idx.values
#
# male_top_color[2] = '000000'
# print(male_top_color)
# l = lambda x: '#'+x
#
# sns.set_style("darkgrid", {"axes.facecolor": "#7C8BA5"})
#
# sns.barplot (x = male_top_sidebar_color, y = male_top_color, palette=list(map(l, male_top_color)))
# plt.show()

#Visualizing colour attributes - sidebar colour (Female)

# female_top_sidebar_color = data[data['gender'] == 'female']['sidebar_color'].value_counts().head(6)
# female_top_sidebar_color_idx = female_top_sidebar_color.index
# female_top_color = female_top_sidebar_color_idx.values
#
# female_top_color[2] = '000000'
# print (female_top_color)
#
# l = lambda x: '#'+x
#
# sns.set_style("darkgrid", {"axes.facecolor": "#7C8BA5"})
# sns.barplot (x = female_top_sidebar_color, y = female_top_color, palette=list(map(l, female_top_color)))
# plt.show()

#Visualizing colour attributes - link colour (male)

# male_top_link_color = data[data['gender'] == 'male']['link_color'].value_counts().head(6)
# male_top_link_color_idx = male_top_link_color.index
# male_top_color = male_top_link_color_idx.values
# male_top_color[1] = '009999'
# male_top_color[5] = '000000'
# print(male_top_color)
#
# l = lambda x: '#'+x
#
# sns.set_style("whitegrid", {"axes.facecolor": "white"})
# sns.barplot (x = male_top_link_color, y = male_top_link_color_idx, palette=list(map(l, male_top_color)))
# plt.show()

#Visualizing colour attributes - link colour (female)

# female_top_link_color = data[data['gender'] == 'female']['link_color'].value_counts().head(7)
# female_top_link_color_idx = female_top_link_color.index
# female_top_color = female_top_link_color_idx.values
#
# l = lambda x: '#'+x
#
# sns.set_style("whitegrid", {"axes.facecolor": "white"})
# sns.barplot (x = female_top_link_color, y = female_top_link_color_idx, palette=list(map(l, female_top_color)))
# plt.show()





#Now we will normalize our data, since our most significant data is in the form of tweets, description (text) ,
# we will use stop words

