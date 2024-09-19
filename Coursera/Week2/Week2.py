import numpy as np
import matplotlib.pyplot as plt

"""
Training Example:
    Size(sqft) | Number of Bedrooms | Number of Floors | Age of Home | Price(1000 dollars)(y)
        2104   |        5           |       1          |       45    |      460 
        1416   |        3           |       2          |       40    |      232
        852    |        2           |       1          |       35    |      178
""" 

#Define Training vector 
X_train = np.array(
                    [
                        [2104, 5, 1, 45],
                        [1416, 3, 2, 40],
                        [852 , 2, 1, 35]
                    ])

y_train = np.array([460, 232, 178])

print(f"X_train.shape = {X_train.shape},  \nX_train = {X_train}")
print(f"y_train.shape = {y_train.shape}, \ny_train = {y_train} ")


#Initialize b and w with small random values
b_init = 785.1811367994083
w_init = np.array([ 0.39133535, 18.75376741, -53.36032453, -26.42131618])
print(f"w_init shape: {w_init.shape}, b_init type: {type(b_init)}")

#Prediction element by element
def predict_single_loop(x,w,b):
    """
    single predict using linear regression
    
    Args:
      x (ndarray): Shape (n,) example with multiple features
      w (ndarray): Shape (n,) model parameters    
      b (scalar):  model parameter     
      
    Returns:
      p (scalar):  prediction
    """
    n = x.shape[0]
    p = 0
    for i in range(0,n):
        p_i = x[i] * w[i]
        p   = p + p_i
    
    p   = p + b
    
    return p

#Use above prediction model
x_vec = X_train[0,:] # Get first row
print(f"x_vec shape = {x_vec.shape}, x_vec = {x_vec}")
#Make a prediction
f_wb = predict_single_loop(x_vec,w_init, b_init)
print(f"f_wb.shape = {f_wb.shape},  predicted house price = {f_wb}")

#Vectorized prediction
def predict(x, w, b): 
    """
    single predict using linear regression
    Args:
      x (ndarray): Shape (n,) example with multiple features
      w (ndarray): Shape (n,) model parameters   
      b (scalar):             model parameter 
      
    Returns:
      p (scalar):  prediction
    """
    p = np.dot(x, w) + b     
    return p    

# get a row from our training data
x_vec = X_train[0,:]
print(f"x_vec shape {x_vec.shape}, x_vec value: {x_vec}")

# make a prediction
f_wb = predict(x_vec,w_init, b_init)
print(f"f_wb shape {f_wb.shape}, prediction: {f_wb}")


#Compute cose
def compute_cost(X, y, w, b): 
    """
    compute cost
    Args:
      X (ndarray (m,n)): Data, m examples with n features
      y (ndarray (m,)) : target values
      w (ndarray (n,)) : model parameters  
      b (scalar)       : model parameter
      
    Returns:
      cost (scalar): cost
    """
    m = X.shape[0]
    cost = 0.0
    for i in range(0,m):
        f_wb_i = np.dot(X[i],w) + b
        cost = cost + (f_wb_i - y_train[i])**2
    cost = cost / (2*m)
    return cost

cost = compute_cost(X_train, y_train, w_init, b_init)
print(f'Cost at optimal w : {cost}')

#COmputer gradient descent with multiple variables
def compute_gradient(X, y, w, b):
    """
    Computes the gradient for linear regression 
    Args:
      X (ndarray (m,n)): Data, m examples with n features
      y (ndarray (m,)) : target values
      w (ndarray (n,)) : model parameters  
      b (scalar)       : model parameter
      
    Returns:
      dj_dw (ndarray (n,)): The gradient of the cost w.r.t. the parameters w. 
      dj_db (scalar):       The gradient of the cost w.r.t. the parameter b. 
    """
    m,n = X.shape
    dj_dw = np.zeros((n,))
    dj_db = 0.

    for i in range(m):
        err = (np.dot(X[i],w) + b) - y[i]
        for j in range(m):
            dj_dw[j] = dj_dw[j] + err*X[i,j]
        dj_db = dj_db + err
    dj_dw = dj_dw / m
    dj_db = dj_db / m

    return dj_dw, dj_db

tmp_dj_dw, tmp_dj_db = compute_gradient(X_train, y_train, w_init, b_init)
print(f'dj_db at initial w,b: {tmp_dj_db}')
print(f'dj_dw at initial w,b: \n {tmp_dj_dw}')

def gradient_descent(X, y, w_in, b_in, cost_function, gradient_function, alpha, num_iters): 
    """
    Performs batch gradient descent to learn w and b. Updates w and b by taking 
    num_iters gradient steps with learning rate alpha
    
    Args:
      X (ndarray (m,n))   : Data, m examples with n features
      y (ndarray (m,))    : target values
      w_in (ndarray (n,)) : initial model parameters  
      b_in (scalar)       : initial model parameter
      cost_function       : function to compute cost
      gradient_function   : function to compute the gradient
      alpha (float)       : Learning rate
      num_iters (int)     : number of iterations to run gradient descent
      
    Returns:
      w (ndarray (n,)) : Updated values of parameters 
      b (scalar)       : Updated value of parameter 
      """
    #J_history = []
