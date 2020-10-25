

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import folium
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
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.svm import SVC
from sklearn.feature_selection import SelectKBest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import f_classif
import numpy as np
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import classification_report
from sklearn.ensemble import VotingClassifier

import datetime
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import minmax_scale
df=pd.read_csv('./data/crime.csv')
print(df.head())

df_code = pd.read_csv('./data/offense_codes.csv')
#df_code.set_index(['OFFENSE_CODE', 'OFFENSE_CODE_EXTENSION'], inplace=True)
#df_code.sort_index(inplace=True)
print(df_code.head())

df['REPORTED_DATE']=df.REPORTED_DATE.apply(lambda x:datetime.datetime.strptime(x,'%m/%d/%Y %I:%M:%S %p'))
df['year']=df.REPORTED_DATE.apply(lambda x:x.strftime('%Y'))
df['month']=df.REPORTED_DATE.apply(lambda x:x.strftime('%m'))
df['day']=df.REPORTED_DATE.apply(lambda x:x.strftime('%d'))
df['hour']=df.REPORTED_DATE.apply(lambda x:x.strftime('%H'))

for col_name in df.columns:
    if(df[col_name].dtype == 'object'):
        df[col_name]= df[col_name].astype('category')
        df[col_name] = df[col_name].cat.codes
df=df.fillna(-999)
df= shuffle(df)
print(df.head())
print("nunique: ",df['OFFENSE_CATEGORY_ID'].nunique())
y=df['OFFENSE_CATEGORY_ID']
print(df.shape)
X=df.drop(['OFFENSE_CATEGORY_ID','FIRST_OCCURRENCE_DATE','LAST_OCCURRENCE_DATE','REPORTED_DATE','INCIDENT_ADDRESS','INCIDENT_ID','OFFENSE_ID','OFFENSE_CODE_EXTENSION'],axis=1)
print('X ',X.shape)
print('Y: ',y.value_counts())

X=minmax_scale(X, feature_range=(0, 1), axis=0, copy=True)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.34,random_state=42,stratify=y)
print('y_test: ', y_test.value_counts())
clf1 = DecisionTreeClassifier(max_depth=7)
clf2 = RandomForestClassifier(n_estimators = 50, criterion = 'entropy', random_state = 0,max_depth=7)
clf3 = LinearDiscriminantAnalysis()
eclf = BaggingClassifier(base_estimator=[('DT', clf1), ('rf', clf2), ('lda', clf3)],max_samples=0.4, max_features=10)

for clf, label in zip([clf1, clf2, clf3, eclf], ['Logistic Regression', 'Random Forest', 'naive Bayes', 'Ensemble']):
    scores = cross_val_score(clf, X, y, cv=10, scoring='accuracy')
    print("Accuracy: %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))
    print("10-fold: ", scores)


kb=StratifiedKFold(10)
y_pred = cross_val_predict(eclf, X, y, cv=kb)
m = accuracy_score(y, y_pred)
print("different accuracy: ", m)
conf_mat = confusion_matrix(y, y_pred)
print(conf_mat)

eclf.fit(X_train, y_train)
y_pred = eclf.predict(X_test)
print(y_pred)
a = accuracy_score(y_test, y_pred)
print('train-test-split accuracy: ', a)
results = confusion_matrix(y_test, y_pred)
print('Confusion Matrix :')
print(results)
print('classification report: ', classification_report(y_test, y_pred))

mse=mean_squared_error(y, y_pred)
print('MSE: ',mse)

#RandomOversample
ros = RandomOverSampler(random_state=42)
X_resampled, y_resampled = ros.fit_resample(X, y)
nb=eclf
kb=StratifiedKFold(10)
scores=cross_val_score(nb,X_resampled,y_resampled,cv=kb,scoring='accuracy')
print("10-fold: ",scores)
print('Decision Tree Random Oversample accuracy: ',scores.mean())
y_pred = cross_val_predict(nb, X_resampled,y_resampled, cv=kb)
conf_mat = confusion_matrix(y_resampled, y_pred)
print(conf_mat)

#RandomUnderSample
rus = RandomUnderSampler(random_state=42)

X_resampled, y_resampled = rus.fit_resample(X, y)
nb=eclf
kb=StratifiedKFold(10)
scores=cross_val_score(nb,X_resampled,y_resampled,cv=kb,scoring='accuracy')
print("10-fold: ",scores)
print('Decision Tree UnderSample accuracy: ',scores.mean())
y_pred = cross_val_predict(nb, X_resampled,y_resampled, cv=kb)
conf_mat = confusion_matrix(y_resampled, y_pred)
print(conf_mat)

#smote for oversample
smote = SMOTE(ratio='minority',random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)
nb=eclf
kb=StratifiedKFold(10)
scores=cross_val_score(nb,X_resampled,y_resampled,cv=kb,scoring='accuracy')
print("10-fold: ",scores)
print('Decision Tree Smote Oversample accuracy: ',scores.mean())
y_pred = cross_val_predict(nb, X_resampled,y_resampled, cv=kb)
conf_mat = confusion_matrix(y_resampled, y_pred)
print(conf_mat)

#tomelinks for undersample
tl = TomekLinks(ratio='majority',random_state=42)
X_resampled, y_resampled  = tl.fit_resample(X, y)

nb=eclf
kb=StratifiedKFold(10)
scores=cross_val_score(nb,X_resampled,y_resampled,cv=kb,scoring='accuracy')
print("10-fold: ",scores)
print('Decision Tree TomeLinks Undersample accuracy: ',scores.mean())
y_pred = cross_val_predict(nb, X_resampled,y_resampled, cv=kb)
conf_mat = confusion_matrix(y_resampled, y_pred)
print(conf_mat)

#balanced
smt = SMOTETomek(ratio='auto',random_state=42)
X_resampled, y_resampled= smt.fit_resample(X, y)
nb=eclf
kb=StratifiedKFold(10)
scores=cross_val_score(nb,X_resampled,y_resampled,cv=kb,scoring='accuracy')
print("10-fold: ",scores)
print('Decision Tree balanced accuracy: ',scores.mean())
y_pred = cross_val_predict(nb, X_resampled,y_resampled, cv=kb)
conf_mat = confusion_matrix(y_resampled, y_pred)
print(conf_mat)



pipeline = Pipeline([
                     ('ANOVA', SelectKBest(f_classif, k='all')),
                     ('clf', eclf)])
# fitting our model and save it in a pickle for later use
model = pipeline.fit(X_train, y_train)

ytest = np.array(y_test)
# confusion matrix and classification report(precision, recall, F1-score)
y_pred=model.predict(X_test)

a=accuracy_score(y_pred,ytest)
print("DecisionTreeClassifier accuracy for SelectKBest: ",a)

print(classification_report(ytest, model.predict(X_test)))
print(confusion_matrix(ytest, model.predict(X_test)))


pipeline = Pipeline([
                     ('VarianceThreshold', VarianceThreshold(threshold=(.8 * (1 - .8)))),
                     ('clf',eclf)])
# fitting our model and save it in a pickle for later use
model = pipeline.fit(X_train, y_train)

ytest = np.array(y_test)
# confusion matrix and classification report(precision, recall, F1-score)
y_pred=model.predict(X_test)

a=accuracy_score(y_pred,ytest)
print("DecisionTreeClassifier accuracy for VarianceThreshold : ",a)

print(classification_report(ytest, model.predict(X_test)))
print(confusion_matrix(ytest, model.predict(X_test)))