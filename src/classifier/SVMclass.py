from sklearn import svm
import DataProcessing as dp

from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

#Creating a classifier for SVM

data=dp.load_data()
JOBS=4
PARAMS=[{ 'C': [4, 2, 1.5, 1, 0.5, 0.1, 0.05, 0.01, 0.001, 0.0001],
           'kernel': ["linear", "poly", "rbf", "sigmoid"],
           'cache_size': [1000],'gamma': ['auto', 1.0, 1.0e-1, 1.0e-2, 1.0e-3, 1.0e-4, 1.0e-5, 1.0e-6]}]
x_train, x_test = dp.split_data()

X_train = dp.extract_feats_from_text(x_train)
X_test=dp.extract_feats_from_text(x_test)
feat_desc_train = dp.extract_feats_from_text_and_desc(x_train)
feat_desc_test = dp.extract_feats_from_text_and_desc(x_test)

y_train,classname_train = dp.encode_class_labels(x_train)

y_test,classname_test = dp.encode_class_labels(x_test)

grid_search=GridSearchCV(SVC(),PARAMS,n_jobs=JOBS,verbose=5,cv=4,scoring="f1")

grid_search.fit(X_train,y_train)

dp.report_results(grid_search, y_train,feat_desc_train, y_test, feat_desc_test, classname_train)


