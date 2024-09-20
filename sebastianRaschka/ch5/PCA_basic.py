#! /usr/bin/env python
import pandas as pd
import numpy  as np

df_wine = pd.read_csv(
                        'https://archive.ics.uci.edu/ml/'
                        'machine-learning-databases/wine/wine.data' ,
                        header=None
                     )

#Load Wine dataset
print(df_wine.shape)

print(df_wine.columns)

from sklearn.model_selection import train_test_split
X, y = df_wine.iloc[:, 1:].values,df_wine.iloc[:,0].values

X_train, X_test, y_train, y_test = \
                                   train_test_split(X, y, test_size=0.3, stratify=y, random_state=0)

#Standardise features
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train_std = sc.fit_transform(X_train)
X_test_std  = sc.fit_transform(X_test)


#Creaer covariance matrix
cov_mat = np.cov(X_train_std.T)
eigen_vals, eigen_vecs = np.linalg.eig(cov_mat)
print(eigen_vals)
tot = sum(eigen_vals)

var_exp = [ ( i / tot) for i in sorted(eigen_vals, reverse=True)]

cum_var_exp = np.cumsum(var_exp)
import matplotlib.pyplot as plt
plt.bar(range(1,14), var_exp, align='center',
        label='Individual explained variance'
       )
plt.step(range(1,14), cum_var_exp, where='mid',
         label='Cummulative explained variance'
        )
plt.ylabel('Explained variance ratio')
plt.xlabel('Principal Component Index')
plt.legend(loc='best')
plt.tight_layout()
plt.show()

eigen_pairs = [(np.abs(eigen_vals[i]), eigen_vecs[:,i]) 
                for i in range(len(eigen_vals)) 
              ]
eigen_pairs.sort(key =lambda k: k[0], reverse=True)

w = np.hstack((eigen_pairs[0][1][:, np.newaxis], eigen_pairs[1][1][:, np.newaxis]))

X_train_pca = X_train_std.dot(w)

colors = ['r', 'b', 'g']
markers = ['o', 's', '^']
for l, c, m in zip(np.unique(y_train), colors, markers) :
    plt.scatter( X_train_pca[y_train==l,0],
                 X_train_pca[y_train==l,1],
                 c=c, label=f'Class {l}', marker=m
               )
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.legend(loc='lower left')
plt.tight_layout()
plt.show()