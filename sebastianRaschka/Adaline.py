import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

class AdalineGD:
    """ADAptive LInear NEuron classifier 
    
    Parameters
    ----------
    eta: float
         Learning rate(between 0.0 to 1.0)
    n_iter: int
        Passes over the training dataset
    
    random_state: int
                  Random NUmber generator seed

    Attributes
    ----------
    w_ : 1-d array
         Weights after fitting
    
    b_ : Scalar
         Bias unit after fitting
    losses_ : int
        Mean squared error loss function after each epoch
    """

    def __init__(self, eta = 0.01, n_iter = 50, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state
    
    def fit(self, X,y):
        """Fit Training data
        Parameters
        ----------
        X: array-like, shape = [n_examples, n_features]
            Trainig vectors, where n_examples is the number of examples and 
            n_features is the number of features
        y: array_like, shape = [n_examples]
        Target values

        Returns
        -------
        self: object
        """

        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=X.shape[1])

        self.b_ = np.float_(0.)
        self.losses_ = []

        for i in range(self.n_iter):
            net_input = self.net_input(X)
            output = self.activation(net_input)
            errors = (y-output)
            self.w_ += self.eta*2.0 * X.T.dot(errors) / X.shape[0]
            self.b_ += self.eta*2.0 * errors.mean()


            loss = (errors**2).mean()
            self.losses_.append(loss)

        return self

    def net_input(self, X):
        return np.dot(X,self.w_) + self.b_
    
    def activation(self,X):
        return X

    def predict(self,X):
        return np.where(self.activation(self.net_input(X))>= 0.5, 1, 0)
    

from matplotlib.colors import ListedColormap
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
print(X)
X_std = np.copy(X)
X_std[:,0] = (X[:,0] - X[:,0].mean()) / X[:,0].std()
X_std[:,1] = (X[:,1] - X[:,1].mean()) / X[:,1].std()

ada_gd = AdalineGD(n_iter=20, eta=0.5)
ada_gd.fit(X_std, y)

plot_decision_regions(X_std, y, classifier=ada_gd)
plt.title('Adaline - Gradient Descent')
plt.xlabel('Sepal Length[std]')
plt.ylabel('Petal Length[std]')
plt.legend(loc='upper left')

plt.tight_layout()

plt.show()

# fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10,4))

# ada1 = AdalineGD(n_iter=15, eta=0.1).fit(X,y)
# ax[0].plot(range(1, len(ada1.losses_)+1), np.log10(ada1.losses_),marker='o')
# ax[0].set_xlabel('Epochs')
# ax[0].set_ylabel('log(MeanSquared error)')
# ax[0].set_title('Adaline - Learning rate 0.1')

# ada2 = AdalineGD(n_iter=15, eta=0.0001).fit(X,y)
# ax[1].plot(range(1, len(ada2.losses_)+1), np.log10(ada2.losses_),marker='o')
# ax[1].set_xlabel('Epochs')
# ax[1].set_ylabel('log(MeanSquared error)')
# ax[1].set_title('Adaline - Learning rate 0.0001')
# plt.show()

