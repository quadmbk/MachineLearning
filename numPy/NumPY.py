import numpy as np
import time

a = np.zeros(4)
print(f"np.zeros(4) = {a}, a.shape(0): {a.shape}, datatype = {a.dtype}")

a = np.zeros((4,1)) #A is a 2 array with 4 roows and 1 column ; shape function output is (rows,columns)
print(f"np.zeros(4) = {a}, a.shape(0): {a.shape}, datatype = {a.dtype}")

a = np.random.random_sample(4) # a is filled with 4 random values
print(f"np.zeros(4) = {a}, a.shape(0): {a.shape}, datatype = {a.dtype}")

a = np.arange(4.) #A gets values from  0 to 4
print(f"np.zeros(4) = {a}, a.shape(0): {a.shape}, datatype = {a.dtype}")

a = np.random.rand(4) #A gets filled with random values
print(f"np.zeros(4) = {a}, a.shape(0): {a.shape}, datatype = {a.dtype}")

a = np.array([5,4,3,2]) #Manually enter integers
print(f"np.array([5,4,3,2]) = {a}, a.shape(0): {a.shape}, datatype = {a.dtype}")

a = np.array([5.,4.,3.,2.]) #Manually enter integers
print(f"np.array([5.,4.,3.,2.]) = {a}, a.shape(0): {a.shape}, datatype = {a.dtype}")

############
# Indexing #
############

# Vector indexing operations on 1-D array

a = np.arange(4)
print(f"np.arange(4) = {a}, a.shape(0): {a.shape}, datatype = {a.dtype}, a[2].shape = {a[2].shape}, a[2] = {a[2]}")

print(f"a[-1] = last element = {a[-1]}")

try:
    c = a[10]
except Exception as e:
    print(f"Index 10 is out of bounds. Exception = {e}")

# Slicing
a = np.arange(10)
print(f"a           = {a}")

#Access 5 consecutive elements (start:stop:step)
c = a[2:7:1]
print(f"a[2:7:1]    = {c}")

#Access 5 consecutive elements separated by two (start:stop:step)
c = a[2:7:2]
print(f"a[2:7:2]    = {c}")

#Access all elements after index 3(index:)
c = a[3:]
print(f"a[3:]       = {c}")

#Access all elements below index 3(:index)
c = a[:3]
print(f"a[:3]       = {c}")

#Access all elements (:)
c = a[:]
print(f"a[:]        = {c}")

# Single Vector operations
a = np.array([1,2,3,4])
print(f"a           : {a}")

# Negate elements of a
b = -a 
print(f"b = -a      : {b}")

# Sum all elements of a. Returns a scalar
b = np.sum(a)
print(f"np.sum(a)   : {b}")

# Mean of a, Returns a scalar
b = np.mean(a)
print(f"np.mean(a)  : {b}")

# SQUARED all elements of a
b = a**2
print(f"a**2        : {b}")

###### VECTOR OPERATIONS #####
a = np.array([1, 2, 3, 4])
b = np.array([-1, -2, 3, 4])

print(f"Binary operators(+,*,-,/,^,&,|) work element wise: {a + b}")
#But this does not work on different size array
c = np.array([1, 2])
try:
    d = a + c
except Exception as e:
    print(f"The Error message is: {e}")

# Scalar vector operations
a = np.array([1,2,3,4])
c = 5 * a
print(f"c = 5 * a : {c}")


# Vector dot product
# Without NumPY
def my_dot(a,b):
    """
        Compute the dot products of two arrays a and b
        Args:
            a = (1-D array) = input vector
            b = (1-D array) = input vector with same dimension as a
        Returns:
            x ( scalar ) 
    """
    x = 0
    for i in range(0,a.shape[0]):
        x += a[i] * b[i]
    
    return x
a = np.arange(4)
print(f"my_dot(a,a) = {my_dot(a,a)}")
print(f"Using np.dot(a,a) = {np.dot(a,a)}")

#Comparing speeds of dot product calculations
np.random.seed()
a = np.random.rand(1000000)
b = np.random.rand(1000000)

tic = time.time()
c = my_dot(a,b)
toc = time.time()
print(f"my_dot(a,b): {c}")
print(f"Loop version duration: {1000*(toc-tic):.4f} ms")

tic = time.time()
c = np.dot(a,b)
toc = time.time()
print(f"np.dot(a,b): {c}")
print(f"Vectorised version duration: {1000*(toc-tic):.4f} ms")
del(a); del(b) #Remove arrays from memory


#### 2-D Array ####

a = np.zeros((1,5)) # zero filled arrays with nrows = 1 and ncols = 5
print(f"a.shape = {a.shape}, nrows = {a.shape[0]}, ncols = {a.shape[1]}, a = {a}")

a = np.zeros((2,1))
print(f"a.shape = {a.shape}, nrows = {a.shape[0]}, ncols = {a.shape[1]}, a = {a}")

a = np.random.random_sample((2,1))
print(f"a.shape = {a.shape}, nrows = {a.shape[0]}, ncols = {a.shape[1]}, a = {a}")


# NumPy routines which allocate memory and fill with user specified values
a = np.array([[5], [4], [3]]);   print(f" a shape = {a.shape}, np.array: a = {a}")
a = np.array([[5],   # One can also
              [4],   # separate values
              [3]]); #into separate rows
print(f" a shape = {a.shape}, np.array: a = {a}")


# Operation of matrices
a = np.arange(6).reshape(-1,2) # -1 tells to comput the number of rows 
                               # given the sized of array and number of columns
                               # Size of array = 6, no of cols = 2, no of rows = 6/2 = 3
print(f"a.shape = {a.shape}, \na = {a}")

# Slicing
a = np.arange(20).reshape(-1,10) # 2D array with size = 20, ncols = 10, nrows = 20/10 = 2
print(f"a.shape = {a.shape}. \n a = {a}")

#access 5 consecutive elements (start:stop:step)
print("a[0, 2:7:1] = ", a[0, 2:7:1], ",  a[0, 2:7:1].shape =", a[0, 2:7:1].shape, "a 1-D array")

#access 5 consecutive elements (start:stop:step) in all rows
print("a[:, 2:7:1] = \n", a[:, 2:7:1], ",  a[:, 2:7:1].shape =", a[:, 2:7:1].shape, "a 2-D array")

# access all elements
print("a[:,:] = \n", a[:,:], ",  a[:,:].shape =", a[:,:].shape)

# access all elements in one row (very common usage)
print("a[1,:] = ", a[1,:], ",  a[1,:].shape =", a[1,:].shape, "a 1-D array")
# same as
print("a[1]   = ", a[1],   ",  a[1].shape   =", a[1].shape, "a 1-D array")
