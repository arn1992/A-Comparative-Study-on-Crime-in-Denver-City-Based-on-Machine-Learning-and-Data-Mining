import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import folium
from sklearn import preprocessing
from sklearn.preprocessing import minmax_scale

from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import TomekLinks
from imblearn.combine import SMOTETomek
#from thundersvm import SVC
from sklearn.model_selection import GridSearchCV
from folium.plugins import HeatMap
from sklearn.model_selection import train_test_split
from statistics import median
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.svm import SVC
from sklearn.feature_selection import SelectKBest

from sklearn.pipeline import Pipeline
from sklearn.feature_selection import f_classif
import numpy as np
from sklearn.metrics import classification_report
df=pd.read_csv('./data/crime.csv')
print(df.head())

df_code = pd.read_csv('./data/offense_codes.csv')
#df_code.set_index(['OFFENSE_CODE', 'OFFENSE_CODE_EXTENSION'], inplace=True)
#df_code.sort_index(inplace=True)
print(df_code.head())



df=df.drop(['FIRST_OCCURRENCE_DATE','LAST_OCCURRENCE_DATE','REPORTED_DATE','INCIDENT_ADDRESS'],axis=1)
print(df.head())
for col_name in df.columns:
    if(df[col_name].dtype == 'object'):
        df[col_name]= df[col_name].astype('category')
        df[col_name] = df[col_name].cat.codes
df=df.fillna(-999)
df= shuffle(df)
print(df.head())
print("nunique: ",df['OFFENSE_CATEGORY_ID'].nunique())
y=df['OFFENSE_CATEGORY_ID']

X=df.drop(['OFFENSE_CATEGORY_ID'],axis=1)
print('X ',X.head())

print('Y: ',y.value_counts())

#X = preprocessing.normalize(X)
X=minmax_scale(X, feature_range=(0, 1), axis=0, copy=True)

X_train,X_test,y_train,y_test=train_test_split(X,y,stratify=y,test_size=.30,random_state=42)

print('y_test: ', y_test.value_counts())
clf= SVC(kernel='linear').fit(X_train,y_train)
clf.fit(X_train,y_train)
y_pred=clf.predict(X_test)
print(y_pred)
a=accuracy_score(y_test,y_pred)
print('train-test-split accuracy: ',a)
results = confusion_matrix(y_test, y_pred)
print ('Confusion Matrix :')
print(results)
print('classification report: ', classification_report(y_test, y_pred))

nb=SVC(kernel='rbf', C=100,gamma='auto')
kb=StratifiedKFold(10)
#kb = StratifiedShuffleSplit(n_splits=10, test_size=0.3, random_state=0)
scores=cross_val_score(nb,X,y,cv=kb,scoring='accuracy')
print("10-fold: ",scores)
print('KNeighborsClassifier accuracy for Labor-relation Dataset: ',scores.mean())
y_pred = cross_val_predict(nb, X,y, cv=kb)


m=accuracy_score(y, y_pred)
print("different accuracy: ", m)
conf_mat = confusion_matrix(y, y_pred)
print(conf_mat)



#RandomOversample
ros = RandomOverSampler(random_state=42)
X_resampled, y_resampled = ros.fit_resample(X, y)
nb=SVC(kernel='rbf', C=100,gamma='auto')
kb=StratifiedKFold(10)
scores=cross_val_score(nb,X_resampled,y_resampled,cv=kb,scoring='accuracy')
print("10-fold: ",scores)
print('SVC Random Oversample accuracy: ',scores.mean())
y_pred = cross_val_predict(nb, X_resampled,y_resampled, cv=kb)
conf_mat = confusion_matrix(y_resampled, y_pred)
print(conf_mat)

#RandomUnderSample
rus = RandomUnderSampler(random_state=42)

X_resampled, y_resampled = rus.fit_resample(X, y)
nb=SVC(kernel='rbf', C=100,gamma='auto')
kb=StratifiedKFold(10)
scores=cross_val_score(nb,X_resampled,y_resampled,cv=kb,scoring='accuracy')
print("10-fold: ",scores)
print('SVC UnderSample accuracy: ',scores.mean())
y_pred = cross_val_predict(nb, X_resampled,y_resampled, cv=kb)
conf_mat = confusion_matrix(y_resampled, y_pred)
print(conf_mat)

#smote for oversample
smote = SMOTE(ratio='minority',random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)
nb=SVC(kernel='rbf', C=100,gamma='auto')
kb=StratifiedKFold(10)
scores=cross_val_score(nb,X_resampled,y_resampled,cv=kb,scoring='accuracy')
print("10-fold: ",scores)
print('SVC Smote Oversample accuracy: ',scores.mean())
y_pred = cross_val_predict(nb, X_resampled,y_resampled, cv=kb)
conf_mat = confusion_matrix(y_resampled, y_pred)
print(conf_mat)

#tomelinks for undersample
tl = TomekLinks(ratio='majority',random_state=42)
X_resampled, y_resampled  = tl.fit_resample(X, y)

nb=SVC(kernel='rbf', C=100,gamma='auto')
kb=StratifiedKFold(10)
scores=cross_val_score(nb,X_resampled,y_resampled,cv=kb,scoring='accuracy')
print("10-fold: ",scores)
print('SVC TomeLinks Undersample accuracy: ',scores.mean())
y_pred = cross_val_predict(nb, X_resampled,y_resampled, cv=kb)
conf_mat = confusion_matrix(y_resampled, y_pred)
print(conf_mat)

#balanced
smt = SMOTETomek(ratio='auto',random_state=42)
X_resampled, y_resampled= smt.fit_resample(X, y)
nb=SVC(kernel='rbf', C=100,gamma='auto')
kb=StratifiedKFold(10)
scores=cross_val_score(nb,X_resampled,y_resampled,cv=kb,scoring='accuracy')
print("10-fold: ",scores)
print('SVC balanced accuracy: ',scores.mean())
y_pred = cross_val_predict(nb, X_resampled,y_resampled, cv=kb)
conf_mat = confusion_matrix(y_resampled, y_pred)
print(conf_mat)



pipeline = Pipeline([
                     ('ANOVA', SelectKBest(f_classif, k='all')),
                     ('clf',SVC(kernel='rbf', C=100,gamma='auto'))])
# fitting our model and save it in a pickle for later use
model = pipeline.fit(X_train, y_train)

ytest = np.array(y_test)
# confusion matrix and classification report(precision, recall, F1-score)
y_pred=model.predict(X_test)

a=accuracy_score(y_pred,ytest)
print("SVC(kernel='linear') accuracy for SelectKBest: ",a)

print(classification_report(ytest, model.predict(X_test)))
print(confusion_matrix(ytest, model.predict(X_test)))


pipeline = Pipeline([
                     ('VarianceThreshold', VarianceThreshold(threshold=(.8 * (1 - .8)))),
                     ('clf', SVC(kernel='rbf', C=100,gamma='auto'))])
# fitting our model and save it in a pickle for later use
model = pipeline.fit(X_train, y_train)

ytest = np.array(y_test)
# confusion matrix and classification report(precision, recall, F1-score)
y_pred=model.predict(X_test)

a=accuracy_score(y_pred,ytest)
print("SVC(kernel='linear') accuracy for VarianceThreshold : ",a)

print(classification_report(ytest, model.predict(X_test)))
print(confusion_matrix(ytest, model.predict(X_test)))

# Grid Search
# Parameter Grid
param_grid = {'C': [0.1, 1, 10, 100], 'gamma': [1, 0.1, 0.01, 0.001, 0.00001, 10]}

# Make grid search classifier
clf_grid = GridSearchCV(SVC(kernel='rbf', C=100,gamma='auto'), param_grid, verbose=1)

# Train the classifier
clf_grid.fit(X_train, y_train)

# clf = grid.best_estimator_()
print("Best Parameters:\n", clf_grid.best_params_)
print("Best Estimators:\n", clf_grid.best_estimator_)
