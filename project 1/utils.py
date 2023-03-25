"""
THE FOLLOWING ARE A SET OF HELPER FUNCTIONS FOR THE closest_pairs ALGORITHM

    External libraries used:
    • math - sqrt function
    • numpy - random number generation
    • matplotlib - plotting

"""

## MATH HELPER FUNCTIONS
import math
import numpy as np

## NOTE THAT bin() AND combinations() ARE FUNCTIONALLY EQUIVALENT

@np.vectorize  ## allows the function to handle scalar AND array inputs
def combinations(n):
    return  np.sum(np.arange(n))

@np.vectorize
def bin(n, k=2):
    return factorial(n) // (factorial(k) * factorial(n-k))

def factorial(n):
    if n == 1:
        return 1
    return n * factorial(n-1)

def distance(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    return math.sqrt(
        (x1-x2)**2 + (y1-y2)**2
    )


## TEST DATA FUNCTIONS
np.random.seed(42)

def sample_p(n):
    return np.random.random(size=(n,2))

def sample_m(n):
    m = combinations(n)
    return np.random.randint(low=1, high=m)

def sample_p_flat(n):
    zeros = np.zeros(shape=(n, 1))
    ints  = np.arange(n)[:,None] #+ np.round(np.random.rand(n,1), 1)
    return np.concatenate((zeros, ints), axis=-1) ** 2
