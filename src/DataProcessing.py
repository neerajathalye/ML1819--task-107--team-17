import random
import re
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

DATA_DIR = "data"


def load_data():
    return pd.read_csv("../dataset/gender-classifier-DFE-791531.csv", encoding='latin1')

def split_data():
    data = pd.read_csv("../dataset/gender-classifier-DFE-791531.csv", encoding='latin1')
    array = ['male', 'female']
    data1 = data.loc[(data['gender:confidence'] == 1) & data['gender'].isin(array)]

    for rows in data1:

        x_train, x_test = train_test_split(data1, test_size=0.2)

    x_train.to_csv('../output-data/train_rows.csv', index=None, header=True)
    x_test.to_csv('../output-data/test_rows.csv', index=None, header=True)
    return x_train, x_test

def encode_class_labels(train_rows, test_rows, df):
    encoder = LabelEncoder()
    y_train = encoder.fit_transform(df.ix[train_rows, "gender"])
    y_test = encoder.transform(df.ix[test_rows, "gender"])

    return y_train, y_test, encoder.classes_

def normalize_text(text):
    # Remove non-ASCII chars.
    text = re.sub('[^\x00-\x7F]+', ' ', text)

    # Remove URLs
    text = re.sub('https?:/.*[\r\n]*', ' ', text)

    # Remove special chars.
    text = re.sub('[?!+%{}:;.,"\'()\[\]_]', '', text)

    # Remove double spaces.
    text = re.sub('\s+', ' ', text)

    return text

df = load_data()

split_data()
