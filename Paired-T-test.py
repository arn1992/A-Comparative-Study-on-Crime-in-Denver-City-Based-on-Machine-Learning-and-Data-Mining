import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import folium
from mlxtend.evaluate import paired_ttest_kfold_cv
from sklearn.preprocessing import minmax_scale
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import TomekLinks
from imblearn.combine import SMOTETomek
from sklearn.utils import shuffle
from folium.plugins import HeatMap
from sklearn.model_selection import train_test_split
from statistics import median
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import VarianceThreshold
import datetime
from sklearn.feature_selection import SelectKBest

from sklearn.pipeline import Pipeline
from sklearn.feature_selection import f_classif
import numpy as np
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
df=pd.read_csv('./data/crime.csv')
df=df[:15000]



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
X=df.drop(['OFFENSE_CATEGORY_ID','FIRST_OCCURRENCE_DATE','LAST_OCCURRENCE_DATE','REPORTED_DATE','INCIDENT_ADDRESS','INCIDENT_ID'],axis=1)
print('X ',X.shape)
print('Y: ',y.value_counts())

X=minmax_scale(X, feature_range=(0, 1), axis=0, copy=True)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.30,random_state=42)


nb1=RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0,max_depth=4)
nb2=nb=DecisionTreeClassifier(max_depth=7)
nb3=LinearDiscriminantAnalysis()

t, p = paired_ttest_kfold_cv(estimator1=nb1,
                              estimator2=nb2,

                              X=X, y=y,
                              random_seed=1)

print('t statistic: %.3f' % t)
print('p value: %.3f' % p)