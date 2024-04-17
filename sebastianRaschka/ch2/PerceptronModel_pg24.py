import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

class Perceptron:
    """Percceptron classifier

    Parameters
    ----------
    eta : float
          Learning rate (between 0.0 and 1.0)
    n_iter : int
             Passes over the training dataset
    random_state : int
                    Random number generator seed for random weight
                    initialisation

    Attributes
    ----------
    w_ : 1d-array
         weight after filtering
    b_ : Scalar
         bias unit after filtering

    errors_ : list
              Number of miscalssifications(updates) in each epoch  
    """

    def __init__(self,eta = 0.1, n_iter = 10, random_state = 1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y):
        """ Fit Training date

            Parameters
            ----------
            X : {array-like}, shape = [n_examples, n_features]
                Training vectors, where n_examples is the number of examples and n_features
                number of features
            y : array-like : shape = [n_examples]
                Target values
            Returns
            -------
            self : object
        """

        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale = 0.01, size = X.shape[1])
        self.b_ = np.float_(0.)
        self.errors_ = []

        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X,y):
                update = self.eta * (target - self.predict(xi))
                self.w_ += update*xi
                self.b_ += update

                errors += int(update != 0.0)
                self.errors_.append(errors)
        return self
    
    def net_input(self,X):
        """Calculate net input"""
        return np.dot(X,self.w_) + self.b_
    
    def predict(self,X):
        """Return class label after unit step"""
        return np.where(self.net_input(X) >= 0.0,1,0) 
    



###MAIN

s = 'https://archive.ics.uci.edu/ml/'\
    'machine-learning-databases/iris/iris.data'


df = pd.read_csv(s, header=None, encoding='utf-8')


#Store expected values in y, following function call, stores 4th column of 0 to 100 rows
y = df.iloc[0:100,4].values

y = np.where(y == 'Iris-setosa',0,1)

#Store features at column 0 and 1 in X, following function call, stores 0th and 1 column of 0 to 100 rows
X = df.iloc[0:100,[0,2]].values
print(df.head())
#plt.scatter(X[:50,0],X[:50,1], color = 'red', marker = 'o', label = 'setosa')
#plt.scatter(X[50:100,0],X[50:100,1], color = 'blue', marker = 's', label = 'Versicolor')

# plt.xlabel('Sepal Length[cm]')
# plt.ylabel('Petal Length[cm]')
# plt.legend(loc='upper left')
# plt.show()



def plot_decision_regions(X,y,classifier, resolution = 0.02):
    markers = ('o', 's', '^', 'v', '<')
    colors  = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap    = ListedColormap(colors[:len(np.unique(y))])

    x1_min, x1_max = X[:,0].min() - 1, X[:,0].max() + 1
    x2_min, x2_max = X[:,1].min() - 1, X[:,1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min,x1_max, resolution), np.arange(x2_min,x2_max,resolution))

    print("X1min: " + str(x1_min) + "x1_max: "+ str(x1_max) + "x2_min: " + str(x2_min) + "x2_max: " + str(x2_max))
    print (np.array([xx1.ravel(), xx2.ravel()]).T)
    lab = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    print("Shubhsha LAB: " + str(xx1.shape)+ " " + str(len(lab)))
    #xx1.shape returns (numrows,num_columns) in xx1 which is (305,235)
    #lab initially is a 1d array of 71675 entries. These are output for all the values xx1, xx2 generated in
    #meshgrid function
    #reshape function divides 71675 as per the shape arguement provided(305,235)
    #Basically Now we have mapping such that (xx1,xx2) can be used as coordinates to access correct y values
    #in lab. This grid can be used to create a filled counter then

    lab = lab.reshape(xx1.shape)
    #print(lab)
    plt.contourf(xx1, xx2, lab, alpha= 0.3, cmap = cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    #np.unique(y) = [0,1]
    #enumerate creates [0,0] and [1,1]
    #This call is used to plot training data
    for idx, c1 in enumerate(np.unique(y)):
        plt.scatter(x= X[y==c1,0],
                    y= X[y==c1, 1],
                    alpha = 0.8,
                    c = colors[idx],
                    marker = markers[idx],
                    label=f'Class{c1}',
                    edgecolor='black' 
                    )
    
ppn = Perceptron(eta = 0.1, n_iter = 10)
ppn.fit(X,y)
plot_decision_regions(X, y, classifier=ppn)
plt.xlabel('Sepal Length[cm]')
plt.ylabel('Petal Length[cm]')
plt.legend(loc='upper left')
plt.show()

#plt.plot(range(1,len(ppn.errors_)+1), ppn.errors_, marker='o')
#plt.xlabel('Epochs')
#plt.ylabel('Number of epochs')
#plt.show()