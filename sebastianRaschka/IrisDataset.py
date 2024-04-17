import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

s = 'https://archive.ics.uci.edu/ml/'\
    'machine-learning-databases/iris/iris.data'

print ('From URL: ', s)

df = pd.read_csv(s, header=None, encoding='utf-8')



#print(df.tail())

y = df.iloc[0:100,4].values
print(y)

y = np.where(y == 'Iris-setosa',0,1)
print(y)

X = df.iloc[0:100,[0,2]].values

plt.scatter(X[:50,0],X[:50,1], color = 'red', marker = 'o', label = 'setosa')
plt.scatter(X[50:100,0],X[50:100,1], color = 'blue', marker = 's', label = 'Versicolor')

plt.xlabel('Sepal Length[cm]')
plt.ylabel('Petal Length[cm]')
plt.legend(loc='upper left')
plt.show()