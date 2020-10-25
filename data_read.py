import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import folium
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
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
df=pd.read_csv('./data/crime.csv')
print(df.head())

df_code = pd.read_csv('./data/offense_codes.csv')
#df_code.set_index(['OFFENSE_CODE', 'OFFENSE_CODE_EXTENSION'], inplace=True)
#df_code.sort_index(inplace=True)
print(df_code.head())

crime=df[['OFFENSE_ID', 'OFFENSE_CODE', 'OFFENSE_CODE_EXTENSION','OFFENSE_TYPE_ID','OFFENSE_CATEGORY_ID']]
crime_1=df.drop(['OFFENSE_CATEGORY_ID','FIRST_OCCURRENCE_DATE','LAST_OCCURRENCE_DATE','REPORTED_DATE','INCIDENT_ADDRESS'],axis=1)

print(crime_1.head())

for col_name in df.columns:
    if(df[col_name].dtype == 'object'):
        df[col_name]= df[col_name].astype('category')
        df[col_name] = df[col_name].cat.codes
df=df.fillna(-999)
print(df.head())
print("nunique: ",df['OFFENSE_CATEGORY_ID'].nunique())
y=df['OFFENSE_CATEGORY_ID']
print('Y: ',y.value_counts())
X=df.drop(['OFFENSE_CATEGORY_ID','FIRST_OCCURRENCE_DATE','LAST_OCCURRENCE_DATE','REPORTED_DATE','INCIDENT_ADDRESS'],axis=1)
print('X ',X.head())


'''
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.40,random_state=42)
clf=RandomForestClassifier(n_estimators = 50, criterion = 'entropy', random_state = 0).fit(X_train,y_train)
clf.fit(X_train,y_train)
y_pred=clf.predict(X_test)
print(y_pred)
a=accuracy_score(y_pred,y_test)
print(a)

nb=RandomForestClassifier(n_estimators = 50, criterion = 'entropy', random_state = 0)
kb=StratifiedKFold(10)
scores=cross_val_score(nb,X,y,cv=10,scoring='accuracy')
print("10-fold: ",scores)
print('Random Forest accuracy for Labor-relation Dataset: ',scores.mean())
y_pred = cross_val_predict(nb, X,y, cv=kb)
conf_mat = confusion_matrix(y, y_pred)
print(conf_mat)


'''
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.40,random_state=42)
clf=GaussianNB().fit(X_train,y_train)
clf.fit(X_train,y_train)
y_pred=clf.predict(X_test)
print(y_pred)
a=accuracy_score(y_pred,y_test)
print(a)

nb=GaussianNB()
kb=StratifiedKFold(10)
scores=cross_val_score(nb,X,y,cv=10,scoring='accuracy')
print("10-fold: ",scores)
print('GaussianNB accuracy for Labor-relation Dataset: ',scores.mean())
y_pred = cross_val_predict(nb, X,y, cv=kb)
conf_mat = confusion_matrix(y, y_pred)
print(conf_mat)

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.40,random_state=42)
clf=KNeighborsClassifier(n_neighbors=5).fit(X_train,y_train)
clf.fit(X_train,y_train)
y_pred=clf.predict(X_test)
print(y_pred)
a=accuracy_score(y_pred,y_test)
print(a)

nb=KNeighborsClassifier(n_neighbors=5)
kb=StratifiedKFold(10)
scores=cross_val_score(nb,X,y,cv=10,scoring='accuracy')
print("10-fold: ",scores)
print('KNeighborsClassifier accuracy for Labor-relation Dataset: ',scores.mean())
y_pred = cross_val_predict(nb, X,y, cv=kb)
conf_mat = confusion_matrix(y, y_pred)
print(conf_mat)

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.40,random_state=42)
clf=DecisionTreeClassifier().fit(X_train,y_train)
clf.fit(X_train,y_train)
y_pred=clf.predict(X_test)
print(y_pred)
a=accuracy_score(y_pred,y_test)
print(a)

nb=DecisionTreeClassifier()
kb=StratifiedKFold(10)
scores=cross_val_score(nb,X,y,cv=10,scoring='accuracy')
print("10-fold: ",scores)
print('DecisionTreeClassifier accuracy for Labor-relation Dataset: ',scores.mean())
y_pred = cross_val_predict(nb, X,y, cv=kb)
conf_mat = confusion_matrix(y, y_pred)
print(conf_mat)
