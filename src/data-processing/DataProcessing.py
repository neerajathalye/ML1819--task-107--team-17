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

# DATA_DIR = "data"
# print(os.getcwd())
dataset_path = os.path.join(os.path.dirname(os.path.dirname(os.getcwd())), 'dataset\\gender-classifier-DFE-791531.csv')
test_rows_path = os.path.join(os.path.dirname(os.path.dirname(os.getcwd())), 'output-data\\test_rows.csv')
train_rows_path = os.path.join(os.path.dirname(os.path.dirname(os.getcwd())), 'output-data\\train_rows.csv')

#df = pd.read_csv(dataset_path, encoding='latin1')
def load_data():
    # print(dataset_path)
    return pd.read_csv(dataset_path, encoding='latin1')

def split_data():

    data = pd.read_csv(dataset_path, encoding='latin1')
    array = ['male', 'female']
    data1 = data.loc[(data['gender:confidence'] == 1) & data['gender'].isin(array)]
    l = len(data1.index)
    indices = np.arange(1, l + 1)
    x_train, x_test ,index_train, index_test = train_test_split(data1, indices, test_size=0.2)

    x_train.to_csv(train_rows_path, index=None, header=True)
    x_test.to_csv(test_rows_path, index=None, header=True)
    return x_train, x_test, index_train, index_test

def encode_class_labels(df):
    encoder = LabelEncoder()
    y_train = encoder.fit_transform(df["gender"])

    return y_train, encoder.classes_

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


def compute_text_feats(vectorizer, rows, df):
    return vectorizer.transform(df.ix[rows, "text_norm"])


# def compute_text_feats_train(vectorizer, df):
#     return vectorizer.transform(df["text_norm"])

def compute_name_feats(vectorizer, df):
    return vectorizer.transform(df["name"])

def compute_text_desc_feats(vectorizer, rows, df):
    text = df.ix[rows, :]["text_norm"]
    desc = df.ix[rows, :]["description_norm"]

    return vectorizer.transform(text.str.cat(desc, sep=' '))

def extract_feats_from_text():

    df = load_data()
    x_train, x_test, index_train, index_test = split_data()
    df["text_norm"] = [normalize_text(text) for text in df["text"]]
    x_train["text_norm"] = [normalize_text(text) for text in x_train["text"]]
    x_test["text_norm"] = [normalize_text(text) for text in x_test["text"]]

    vectorizer = CountVectorizer()

    vectorizer = vectorizer.fit(df.ix[index_train, :]["text_norm"])

    x_tr = compute_text_feats(vectorizer, index_train, df)
    x_te = compute_text_feats(vectorizer, index_test, df)

    return x_tr, x_te

def extract_feats_from_name(df):
    df["name_norm"] = [normalize_text(text) for text in df["name"]]

    vectorizer = CountVectorizer()
    vectorizer = vectorizer.fit(df["name_norm"])

    X = compute_name_feats(vectorizer, df)

    return X


def extract_feats_from_text_and_desc():

    df = load_data()
    x_train, x_test, index_train, index_test = split_data()

    df["text_norm"] = [normalize_text(text) for text in df["text"]]
    df["description_norm"] = [normalize_text(text) for text in df["description"].fillna("")]
    x_train["text_norm"] = [normalize_text(text) for text in x_train["text"]]
    x_train["description_norm"] = [normalize_text(text) for text in x_train["description"].fillna("")]
    x_test["text_norm"] = [normalize_text(text) for text in x_test["text"]]
    x_test["description_norm"] = [normalize_text(text) for text in x_test["description"].fillna("")]

    vectorizer = CountVectorizer()

    text = df.ix[index_train, :]["text_norm"]
    desc = df.ix[index_train, :]["description_norm"]
    vectorizer = vectorizer.fit(text.str.cat(desc, sep=' '))

    x_tr = compute_text_desc_feats(vectorizer, index_train, df)
    x_te = compute_text_desc_feats(vectorizer, index_test , df)

    return x_tr, x_te

def extract_tweet_count_feats(df):
    feats = df[["retweet_count", "tweet_count", "fav_number"]]

    scaler = StandardScaler().fit(feats)

    return scaler.transform(feats)

def print_results(y_true, y, X, data_set_name, class_names):
    print(data_set_name)
    print(classification_report(y, y_true, target_names=class_names))
    print("Accuracy: {}".format(accuracy_score(y, y_true)))
    print("==================================================================")
    print()


def report_results(grid_search, y_train, X_train, y_test, X_test, class_names):
    print("Best params: ", grid_search.best_params_)
    print_results(grid_search.predict(X_train), y_train, X_train, "Train", class_names)
    print_results(grid_search.predict(X_test), y_test, X_test, "Test", class_names)


df = load_data()

split_data()
