from sklearn import svm
import DataProcessing as dp

from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

#Creating a classifier for SVM

data=dp.load_data()
JOBS=4
PARAMS=[{ 'kernel': ["poly"],
           'cache_size': [100],'gamma': ['auto', 1.0, 1.0e-1, 1.0e-2, 1.0e-3, 1.0e-4, 1.0e-5, 1.0e-6]}]

x_train, x_test,index_train, index_test = dp.split_data()

y_train,classname_train = dp.encode_class_labels(x_train)

y_test,classname_test = dp.encode_class_labels(x_test)

print ("features from text")
X_train,X_test = dp.extract_feats_from_text()
#X_test=dp.extract_feats_from_text()

grid_search=GridSearchCV(SVC(),PARAMS,n_jobs=JOBS,verbose=5,cv=4,scoring="f1")

grid_search.fit(X_train,y_train)
dp.report_results(grid_search, y_train,X_train, y_test, X_test, classname_train)    #prints features wiht names

print("features from text and description")
feat_desc_train,feat_desc_test = dp.extract_feats_from_text_and_desc()


grid_search=GridSearchCV(SVC(),PARAMS,n_jobs=JOBS,verbose=5,cv=4,scoring="f1")

grid_search.fit(feat_desc_train,y_train)

dp.report_results(grid_search, y_train,feat_desc_train, y_test,feat_desc_test, classname_train)   #prints features with name and description
