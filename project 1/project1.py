import numpy as np
from tqdm import trange
from copy import deepcopy

import utils


## DATA STRUCTURES

class Node:
    def __init__(self, dist:float, p1:tuple, p2:tuple):
        self.dist = dist
        self.p1 = p1
        self.p2 = p2

    ## OVERLOAD GREATER THAN OPERATOR
    def __gt__(self, other):
        return self.dist > other.dist


class MinHeap:

    ## ARRAY BASED IMPLEMENTATION
    def __init__(self, m) -> None:
        # self.size = 0
        self.array = []
        self.comparisons = 0
        self.max_depth = int(np.log2(m))

    def insert(self, node):
        start = self.comparisons

        ## ADD NEW NODE TO BOTTOM OF TREE
        self.array.append(node)
        
        # self.size += 1
        ## HEAPIFY
        self.heapify_up(self.size-1)
        
        ## DROP NODES BEYOND THE MAX DEPTH
        # if self.depth > self.max_depth:
        #     self.array = self.array[:self.size-1]
        #     # self.size -= 1
        print(self.comparisons-start)

    def get_min(self):
        start = self.comparisons
        if self.empty():
            raise Exception("ERROR: Can't get min from an empty heap!")
        
        node = deepcopy(self.array[0])

        # self.size -= 1
        ## SWAP LAST NODE AND ROOT
        self.array[0] = self.array[self.size-1]

        ## REMOVE LAST NODE FROM TREE
        self.array = self.array[:self.size-1]

        ## HEAPIFY 
        self.heapify_down()

        print(self.comparisons-start)

        return node

    # ## GENERIC HEAPIFY COMPARISON AND SWAP
    # def heapify(self, parent_idx, child_idx, mode="up"):
    #     ## HALT WHEN LEAF OR ROOT IS SWAPPED
    #     if child_idx < self.size and parent_idx >= 0:
    #         if self.compare(parent_idx, child_idx):
    #             self.swap(parent_idx, child_idx)
    #             ## MODE DETERMINES DIRECTION OF RECURSION
    #             if mode == "up":
    #                 return self.heapify_up(parent_idx)
    #             elif mode == "down":
    #                 return self.heapify_down(child_idx)
    #     #     else:
    #     #         return True
    #     # else:
    #     #     return True
                
    def heapify_down(self, parent_idx=0):
        ## GET CHILD NODE INDICES
        left_idx = 2*parent_idx + 1
        right_idx = 2*parent_idx + 2

        ## FIND SMALLEST CHILD
        if right_idx < self.size and self.compare(left_idx, right_idx):
            child_idx = right_idx
        elif left_idx < self.size:
            child_idx = left_idx
        ## NO CHILDREN
        else:
            return

        ## SWAP IF PARENT IS GREATER THAN SMALLEST CHILD
        if self.compare(parent_idx, child_idx):
            self.swap(parent_idx, left_idx)
        ## OTHERWISE RETURN (BASE CASE)
        else:
            return

        return self.heapify_down(child_idx)
        # ## is left child smaller? 
        # ##  • swap
        # ##  • heapify down left child index
        # ## is right child smaller
        # ##  • swap
        # ##  • heapify down right child index

        # return self.heapify(idx, left_idx, "down") \
        #     or self.heapify(idx, right_idx, "down")
        
    def heapify_up(self, child_idx):
        ## GET PARENT NODE INDEX
        parent_idx = (child_idx-1)//2 if child_idx % 2 else (child_idx-2)//2

        ## SWAP IF PARENT IS GREATER THAN CHILD
        if parent_idx >= 0 and self.compare(parent_idx, child_idx):
            self.swap(parent_idx, child_idx)
            child_idx = parent_idx
        ## OTHERWISE RETURN (BASE CASE)
        else:
            return
        return self.heapify_up(child_idx)

        # return self.heapify(parent_idx, idx, "up")

    def compare(self, parent_idx, child_idx):
        ## RETURN TRUE IF PARENT INDEX IS GREATER THAN CHILD
        ## (this indicates that a swap is required)
        self.comparisons += 1
        return self.array[parent_idx] > self.array[child_idx]

    def swap(self, parent_idx, child_idx):
        temp = deepcopy(self.array[parent_idx])
        self.array[parent_idx] = self.array[child_idx]
        self.array[child_idx] = temp

    def empty(self):
        return self.size <= 0

    @property
    def size(self):
        return len(self.array)

    @property
    def depth(self):
        return int(np.log2(self.size))

## ALGORITHM

def closest_pairs(p, m):
    ## INIT HEAP
    heap = MinHeap(m)

    comparisons = 0

    ## CHECK m
    n = len(p)
    assert m <= utils.combinations(n), "ERROR: m is greater than the possible number of combinations of points given"

    ## COMPUTE DIST BETWEEN EACH COMBINATION OF POINTS
    ## T(n) = \sum(n-1) -> M lg M  operation per step
    ## np.sum(np.arange(10))
    for i in trange(n):  # O(M log2 M) -> O(M log2 m) -> 

        for j in range(i+1,n):

            ## GET DIST
            dist = utils.distance(p[i], p[j])
            ## TRACK COMPARISONS
            comparisons += 1
            ## CONSTRUCT NODE
            node = Node(dist, p[i], p[j])
            ## INSERT NODE INTO HEAP
            heap.insert(node) ## TODO: DISCARD ANY NODE BELOW LEVEL M IN THE HEAP

    ## GET THE M CLOSEST PAIRS FROM THE HEAP
    pairs = []
    ## T(n) = m -> log m operations per step
    for i in trange(m): # O(m log2 M) -> O(m log2 m)
        node = heap.get_min()  ## log2 m
        # pairs.append((node.p1, node.p2))
        pairs.append(node)

    comparisons += heap.comparisons
    return pairs, comparisons


## ANALYSYS
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def test_worst_case():
    cases = range(4,50)
    comparisons = []
    for n in cases:
        p = utils.sample_p(n)
        m = utils.combinations(n)
        _, k = closest_pairs(p, m)
        comparisons.append(k)

    fig = plt.figure(figsize=(4, 3), dpi=300)
    ax = fig.add_subplot(111)

    cases = np.array(cases)
    program = ax.plot(cases, comparisons, color="orange", label="Program Time")
    theory = ax.plot(cases, cases**2, color="red", label="Theoretical Time")
        
    fig.suptitle("Time Complexity Analysis", fontsize=10, fontweight='normal', alpha=0.5, y=0.96)
    ax.set_ylabel("Comparisons", fontweight="normal", alpha=0.5, fontsize="x-small")
    ax.set_xlabel("Number of Points", fontweight="normal", alpha=0.5, fontsize="x-small")
    ax.tick_params(axis='both', which='major', labelsize=3)
    ax.legend(loc=1, prop={'size': 4})


    fig.canvas.draw()
    fig.savefig("worst_case.png")


    return
    






'''
m is significant because its maximum value b = bin(n,2) represents the maximum
number of unique pairs of points in our dataset p. We must loop through at least
b pairs in order to solve the problem. Looping any more than b times would be wasteful. 

Since our heap insert method leads to full levels at each depth of the heap,
once we fill enough levels to capture hold at least m elements, we can drop
nodes that are heapified down beyond that depth. This keeps our tree at a
max depth of log2(m) as opposed to log2(n*c), where c = bin(n, 2) and
1 ≤ m ≤ c. 


numpy used for logarithm, sqrt, arange, and random

'''



if __name__ == "__main__":

    import argparse

    21_898_889

    n = 10
    # n = 100
    # p = utils.sample_p(n)
    p = utils.sample_p_flat(n)
    # m = utils.sample_m(n)
    m = utils.combinations(n)
    # m = 10

    ans, comp = closest_pairs(p, m)
    t = [a.dist for a in ans]

    test_worst_case()

    test = MinHeap()







