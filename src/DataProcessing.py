import random
import re
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

DATA_DIR = "data"


def load_data():
    return pd.read_csv("../dataset/gender-classifier-DFE-791531.csv", encoding='latin1')

def select_rows(df):
    return df[df["gender"].isin(["male", "female"]) &
              (df["gender:confidence"] > 0.99)].index.tolist()

def split_data(rows):
    n_samples = len(rows)
    random.shuffle(rows)
    test_size = round(n_samples * 0.2)
    test_rows = rows[:test_size]
    train_rows = rows[test_size:]
    m = len(test_rows)
    n = len(train_rows)
    df1 = pd.DataFrame(np.array(test_rows).reshape(m,1))
    df2 = pd.DataFrame(np.array(train_rows).reshape(n,1))
    df1.to_csv ('../output-data/test_rows.txt', index = None, header=True)
    df2.to_csv ('../output-data/train_rows.txt', index = None, header=True)

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

def load_data_split():
    train = pd.read_csv("{}/train_rows.txt".format(DATA_DIR)).row_number.tolist()
    test = pd.read_csv("{}/test_rows.txt".format(DATA_DIR)).row_number.tolist()

    return train, test

df = load_data()
selected_rows = select_rows(df)

split_data(selected_rows)