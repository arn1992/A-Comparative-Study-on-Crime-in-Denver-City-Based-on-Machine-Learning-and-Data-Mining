import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
import scipy.cluster.hierarchy as shc
from sklearn.cluster import AgglomerativeClustering


data = pd.read_csv('D:/machine learning fall2019/final_project/Li-ion-hierarchical-clustering-master 2/Li-ion-hierarchical-clustering-master/Liion_comp_528.csv')
print(data.head())
#data = data[(data.T != 0).any()]
#print(data.shape)

data = data.iloc[:, 4:].values
#data=normalize(data)
print(data)

plt.figure(figsize=(10, 7))
plt.title("Customer Dendograms")
dend = shc.dendrogram(shc.linkage(data, method='ward'))
plt.show()

'''cluster = AgglomerativeClustering(n_clusters=7, affinity='euclidean', linkage='ward')
cluster.fit_predict(data)
print(cluster.labels_)'''


model = AgglomerativeClustering(n_clusters=7, affinity='cosine', linkage='single')
model.fit_predict(data)
labels = model.labels_
print(labels)


plt.figure(figsize=(10, 7))
plt.xlim(-.0000001, .0000001)
plt.ylim(-.000001, .0000001)
plt.scatter(data[labels==0, 0], data[labels==0, 1], s=50, marker='o', color='red')
plt.scatter(data[labels==1, 0], data[labels==1, 1], s=50, marker='+', color='blue')
plt.scatter(data[labels==2, 0], data[labels==2, 1], s=50, marker='1', color='green')
plt.scatter(data[labels==3, 0], data[labels==3, 1], s=50, marker='2', color='purple')
plt.scatter(data[labels==4, 0], data[labels==4, 1], s=50, marker='3', color='orange')
plt.scatter(data[labels==5, 0], data[labels==5, 1], s=50, marker='3', color='black')
plt.scatter(data[labels==6, 0], data[labels==6, 1], s=50, marker='3', color='brown')
#plt.scatter(data[labels==7, 0], data[labels==7, 1], s=50, marker='3', color='gold')
#plt.scatter(data[labels==8, 0], data[labels==8, 1], s=50, marker='3', color='crimson')
#plt.scatter(data[labels==9, 0], data[labels==9, 1], s=50, marker='3', color='darkred')
plt.show()
