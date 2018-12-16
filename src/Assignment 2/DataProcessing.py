# Import libraries

import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import re
import os.path
from collections import Counter
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB

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

data.drop (columns = ['sidebar_color'], inplace = True)

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

tweet_vocab = Counter()
for tweet in data['text']:
    for word in tweet.split(' '):
        tweet_vocab[word] += 1

#Printing the most common 30 words
# print(tweet_vocab.most_common(30))

nltk.download('stopwords')

stop = stopwords.words('english')

tweet_vocab_reduced = Counter()
for i, j in tweet_vocab.items():
    if not i in stop:
        tweet_vocab_reduced[i]=j

#Printing reduced vocabulory
# print(tweet_vocab_reduced.most_common(30))

#Text Clean Function

def TextClean(text):
    """ Return a cleaned version of text
    """
    # Remove HTML markup
    text = re.sub('<[^>]*>', '', text)
    # Save emoticons for later appending
    emojis = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text)
    # Remove any non-word character and append the emoticons,
    # removing the nose character for standarization. Convert to lower case
    text = (re.sub('[\W]+', ' ', text.lower()) + ' ' + ' '.join(emojis).replace('-', ''))

    return text


#print(TextClean('This!!@ twit :) is <b>nice</b>'))

#Further normalization using Porter Stemming

porter = PorterStemmer()

def tokenizer(text):
    return text.split()

def porter_tokenizer(text):
    return [porter.stem(word) for word in text.split()]

#print(tokenizer('Hi there, I am loving this, like with a lot of love running bunning shunning cunning fucking harder runs funs guns ponies'))
#print(porter_tokenizer('Hi there, I am loving this, like with a lot of love running bunning shunning cunning fucking harder runs funs guns ponies'))

#Encoding and Splitting the data set into test and train


encoder = LabelEncoder()
y = encoder.fit_transform(data['gender']) # encode male = 1, female = 0
# print(data['gender'])
# print(y)


def MLAlgorithms():

    tfidf = TfidfVectorizer(lowercase=False,
                            tokenizer=porter_tokenizer,
                            preprocessor=TextClean)

    JOBS = 4

    # Logistic Regression

    PARAMS = [{'penalty': ["l1", "l2"], 'C': [10, 5, 1, 0.5, 0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001]}]

    clf = Pipeline([('vect', tfidf),
                    ('clf', GridSearchCV(LogisticRegression(), PARAMS, n_jobs=JOBS, verbose=5, cv=4, scoring="f1"))])

    clf.fit(X_train, y_train)

    print("----------------------------------------------------------------")
    print("LOGISTIC REGRESSION")
    predictions = clf.predict(X_test)
    print('Accuracy:', accuracy_score(y_test, predictions))
    print('Confusion matrix:\n', confusion_matrix(y_test, predictions))
    print('Classification report:\n', classification_report(y_test, predictions))

    # Naive Bayes

    PARAMS = [{'alpha': [10, 5, 1, 0.5, 0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001]}]

    clf = Pipeline([('vect', tfidf),
                    ('clf', GridSearchCV(MultinomialNB(), PARAMS, n_jobs=JOBS, verbose=5, cv=4,
                                         scoring="f1"))])

    clf.fit(X_train, y_train)

    print("----------------------------------------------------------------")
    print("NAIVE BAYES")

    predictions = clf.predict(X_test)
    print('Accuracy:', accuracy_score(y_test, predictions))
    print('Confusion matrix:\n', confusion_matrix(y_test, predictions))
    print('Classification report:\n', classification_report(y_test, predictions))

    # SVM

    PARAMS = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],'C': [1, 10, 100, 1000]}, {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]

    clf = Pipeline([('vect', tfidf),
                    ('clf', GridSearchCV(SVC(), PARAMS, n_jobs=JOBS, verbose=5, cv=4, scoring="f1"))])
    clf.fit(X_train, y_train)
    print("----------------------------------------------------------------")
    print("SUPPORT VECTOR MACHINE")
    predictions = clf.predict(X_test)
    print('Accuracy:', accuracy_score(y_test, predictions))
    print('Confusion matrix:\n', confusion_matrix(y_test, predictions))
    print('Classification report:\n', classification_report(y_test, predictions))



# split the dataset in train and test(FOR text)
X = data['text']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)
#In the code line above, stratify will create a train set with the same class balance than the original set
# print(X_train.head())

print("USING TEXT")
MLAlgorithms()


#Using Text+Description

data.fillna("", inplace = True) # Replacing all NaN with empty strings
data['text_description'] = data['text'].str.cat(data['description'], sep=' ') #Concatenate the text and descriptions into a new column text_description

# print(data['text_description'].isnull().value_counts()) #Check if there are any null values in this column

X = data['text_description']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)
#In the code line above, stratify will create a train set with the same class balance than the original set
# X_train.head()
# print(X_train.isnull().values.any()) # Check if any null values, True if there is at least one.
print("USING TEXT AND DESCRIPTION")
MLAlgorithms()









