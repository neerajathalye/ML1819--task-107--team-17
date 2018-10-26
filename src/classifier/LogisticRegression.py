import pandas as pd
import os.path
import matplotlib.pyplot as plt
import seaborn as sns

plt.rc("font", size=14)
sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)


datasetPath = os.path.join(os.path.dirname(os.path.dirname(os.getcwd())), 'output-data\\train_rows.csv')

print(datasetPath)

data = pd.read_csv(datasetPath, header=0, encoding='latin1')
print(data.shape)
print(list(data.columns))
# print(data.head())
#
# print(data['gender'].value_counts())
#
# # To see columns with missing values
# print(data.isnull().sum())
# # print(data.info())
#
# # Get the columns for description and name
# gender_data = data.loc[:, ('description', 'name')].values
# gender_data_names = ['description', 'names']
#
# # Get the column for gender
# y = data.loc[:, 'gender'].values
#
# X = scale(gender_data)
#
# LogReg = LogisticRegression()
#
# LogReg.fit(X, y)
#
# print(LogReg.score(X, y))
#
# y_pred = LogReg.predict(X)

# print(classification_report((y, y_pred)))














# sns.countplot(x='gender', data=data, palette='hls')
# plt.show()

