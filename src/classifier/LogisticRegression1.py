import os.path
from DataProcessing import load_data
from DataProcessing import split_data
from DataProcessing import encode_class_labels
from DataProcessing import report_results
from DataProcessing import extract_feats_from_text
from DataProcessing import extract_feats_from_name
from DataProcessing import extract_feats_from_text_and_desc
from DataProcessing import extract_feats_from_text_and_desc_and_name
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

import matplotlib.pyplot as plt
import seaborn as sns

plt.rc("font", size=14)
sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)

PARAMS = [{'penalty': ["l1", "l2"], 'C': [4, 2, 1.5, 1, 0.5, 0.1, 0.05, 0.01, 0.001, 0.0001]}]
JOBS = 4

data = load_data()
# print(list(data.columns))

x_train, x_test , index_train1, index_test1 = split_data()

# print(data.shape)
# print(x_test.shape)
# print(x_train.shape)
# array = ['male', 'female']
# data1 = x_test.loc[:, 'gender'].values
# data2 = x_train.loc[:, 'gender'].values

y_train, class_names_train = encode_class_labels(x_train)
y_test, class_names_test = encode_class_labels(x_test)


# print(len(data1))
# print(len(data2))

# print(data1)
# print(data2)

# print(y_test)
# print(y_train)

# X_train, X_test = extract_feats_from_text()
X_train, X_test = extract_feats_from_text_and_desc()
# X_train, X_test = extract_feats_from_text_and_desc_and_name()

grid_search = GridSearchCV(LogisticRegression(), PARAMS, n_jobs=JOBS, verbose=5, cv=4, scoring="f1")

grid_search.fit(X_train, y_train)

report_results(grid_search, y_train, X_train, y_test, X_test, class_names_train)











