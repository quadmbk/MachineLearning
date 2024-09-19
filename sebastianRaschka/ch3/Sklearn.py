import numpy as np
from sklearn import datasets
#To randomly split data for training and testing
from sklearn.model_selection import train_test_split

#To normalise data
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import Perceptron

s = 'https://archive.ics.uci.edu/ml/'\
    'machine-learning-databases/iris/iris.data'

iris = datasets.load_iris()



X = iris.data[:,[2,3]]
y = iris.target

print("Class Targets", np.unique(y))

"""
    test_size=0.3 //Means split X and y into 30 percent test data and 0% trainnig data
    random_state=1, the random seed
    stratify=y //train and test data shall have equal proportions of class labels and input dataset
"""
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=1, stratify=y)

print("Labels count in y:", np.bincount(y))
print("Labels in y_train:",np.bincount(y_train))
print("Labels in y_test:",np.bincount(y_test))


sc = StandardScaler()
sc.fit(X_train) #Calculates the mean and SD over X_train data
X_train_std = sc.transform(X_train)
X_test_std  = sc.transform(X_test) #WE use mean and SD so, that both X_train and X_test are comparable

ppn = Perceptron(eta0=0.1, random_state=1)
ppn.fit(X_train_std, y_train)
y_pred = ppn.predict(X_test_std)
print(y_pred)
print(y_test)
mis = (y_test!=y_pred).sum()
print("Miscalssifed examples: %d" % mis)
accuracy = (1 - (mis)/y_test.shape[0]) * 100
print("Accuracy: %f \\%" % accuracy)
