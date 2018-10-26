import re
import os.path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import CountVectorizer

DATA_DIR = "data"
# print(os.getcwd())
dataset_path = os.path.join(os.path.dirname(os.path.dirname(os.getcwd())), 'dataset\\gender-classifier-DFE-791531.csv')
test_rows_path = os.path.join(os.path.dirname(os.path.dirname(os.getcwd())), 'output-data\\test_rows.csv')
train_rows_path = os.path.join(os.path.dirname(os.path.dirname(os.getcwd())), 'output-data\\train_rows.csv')


def load_data():
    # print(dataset_path)
    return pd.read_csv(dataset_path, encoding='latin1')

def split_data():

    data = pd.read_csv(dataset_path, encoding='latin1')
    array = ['male', 'female']
    data1 = data.loc[(data['gender:confidence'] == 1) & data['gender'].isin(array)]

    x_train, x_test = train_test_split(data1, test_size=0.2)

    x_train.to_csv(train_rows_path, index=None, header=True)
    x_test.to_csv(test_rows_path, index=None, header=True)
    return x_train, x_test

def encode_class_labels_train(df):
    encoder = LabelEncoder()
    y_train = encoder.fit_transform(df["gender"])

    return y_train, encoder.classes_

def encode_class_labels_test(df):
    encoder = LabelEncoder()
    y_test = encoder.transform(df["gender"])

    return y_test, encoder.classes_

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


def compute_text_feats_test(vectorizer, df):
    return vectorizer.transform(df["text_norm"])


def compute_text_feats_train(vectorizer, df):
    return vectorizer.transform(df["text_norm"])

def compute_name_feats_test(vectorizer, df):
    return vectorizer.transform(df["name"])


def compute_name_feats_train(vectorizer, df):
    return vectorizer.transform(df["name"])


def compute_text_desc_feats_test(vectorizer, df):
    train_text = df["text_norm"]
    train_desc = df["description_norm"]

    return vectorizer.transform(train_text.str.cat(train_desc, sep=' '))


def compute_text_desc_feats_train(vectorizer, df):
    train_text = df["text_norm"]
    train_desc = df["description_norm"]

    return vectorizer.transform(train_text.str.cat(train_desc, sep=' '))


def extract_feats_from_text_test(df):
    df["text_norm"] = [normalize_text(text) for text in df["text"]]
    # df["description_norm"] = [normalize_text(text) for text in df["description"].fillna("")]

    vectorizer = CountVectorizer()
    vectorizer = vectorizer.fit(df["text_norm"])

    X_test = compute_text_feats_test(vectorizer, df)

    return X_test


def extract_feats_from_text_train(df):
    df["text_norm"] = [normalize_text(text) for text in df["text"]]
    # df["description_norm"] = [normalize_text(text) for text in df["description"].fillna("")]

    vectorizer = CountVectorizer()
    vectorizer = vectorizer.fit(df["text_norm"])

    X_train = compute_text_feats_test(vectorizer, df)

    return X_train

def extract_feats_from_name_test(df):
    df["name_norm"] = [normalize_text(text) for text in df["name"]]
    # df["description_norm"] = [normalize_text(text) for text in df["description"].fillna("")]

    vectorizer = CountVectorizer()
    vectorizer = vectorizer.fit(df["name_norm"])

    X_test = compute_text_feats_test(vectorizer, df)

    return X_test


def extract_feats_from_name_train(df):
    df["name_norm"] = [normalize_text(text) for text in df["name"]]
    # df["description_norm"] = [normalize_text(text) for text in df["description"].fillna("")]

    vectorizer = CountVectorizer()
    vectorizer = vectorizer.fit(df["name_norm"])

    X_train = compute_text_feats_test(vectorizer, df)

    return X_train


def extract_feats_from_text_and_desc_test(df):
    df["text_norm"] = [normalize_text(text) for text in df["text"]]
    df["description_norm"] = [normalize_text(text) for text in df["description"].fillna("")]

    vectorizer = CountVectorizer()
    train_text = df["text_norm"]
    train_desc = df["description_norm"]
    vectorizer = vectorizer.fit(train_text.str.cat(train_desc, sep=' '))

    X_test = compute_text_desc_feats_test(vectorizer, df)

    return X_test


def extract_feats_from_text_and_desc_train(df):
    df["text_norm"] = [normalize_text(text) for text in df["text"]]
    df["description_norm"] = [normalize_text(text) for text in df["description"].fillna("")]

    vectorizer = CountVectorizer()
    train_text = df["text_norm"]
    train_desc = df["description_norm"]
    vectorizer = vectorizer.fit(train_text.str.cat(train_desc, sep=' '))

    X_train = compute_text_desc_feats_test(vectorizer, df)

    return X_train


def extract_tweet_count_feats_test(df):
    feats_test = df[["retweet_count", "tweet_count", "fav_number"]]

    scaler = StandardScaler().fit(feats_test)

    return (scaler.transform(feats_test))


def extract_tweet_count_feats_train(df):
    feats_train = df[["retweet_count", "tweet_count", "fav_number"]]

    scaler = StandardScaler().fit(feats_train)

    return (scaler.transform(feats_train))

def print_results(y_true, y, data_set_name, class_names):
    print(data_set_name)
    print(classification_report(y, y_true, target_names=class_names))
    print("Accuracy: {}".format(accuracy_score(y, y_true)))
    print("==================================================================")
    print()


def report_results(grid_search, y_train, X_train, y_test, X_test, class_names):
    print("Best params: ", grid_search.best_params_)
    print_results(grid_search.predict(X_train), y_train, "Train", class_names)
    print_results(grid_search.predict(X_test), y_test, "Test", class_names)


df = load_data()

split_data()
