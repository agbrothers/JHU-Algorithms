import numpy as np
from tqdm import trange

import utils


DEBUG = False

## DATA STRUCTURES

class Node:
    def __init__(self, dist:float, p1:tuple=None, p2:tuple=None, idx:int=None):
        self.dist = dist
        self.p1 = p1
        self.p2 = p2
        self.idx = idx

    ## OVERLOAD GREATER THAN OPERATOR
    def __gt__(self, other):
        return self.dist > other.dist
    
    ## OVERLOAD LESS THAN OPERATOR
    def __lt__(self, other):
        return self.dist < other.dist


class MinHeap:

    ## ARRAY BASED IMPLEMENTATION
    def __init__(self, m) -> None:
        self.array = []
        self.comparisons = 0
        self.max_depth = int(np.log2(m))

    def insert(self, node):
        if DEBUG: start = self.comparisons
        ## ADD NEW NODE TO BOTTOM OF TREE
        self.array.append(node)
        ## HEAPIFY
        self.heapify_up(self.size-1)
        ## DROP NODES BEYOND THE MAX DEPTH
        if self.depth > self.max_depth:
            self.array = self.array[:self.size-1]
        if DEBUG: print(self.comparisons-start)

    def get_min(self):
        if DEBUG: start = self.comparisons
        if self.empty(): raise Exception("ERROR: Can't get min from an empty heap!")
        ## RETURN ROOT NODE (MINIMUM)
        node = self.array[0]
        ## DEBUG CHECK
        # if DEBUG: assert node.dist == min([i.dist for i in self.array])
        ## SWAP LAST NODE AND ROOT
        self.array[0] = self.array[self.size-1]
        ## REMOVE LAST NODE FROM TREE
        self.array = self.array[:self.size-1]
        ## HEAPIFY 
        self.heapify_down()
        if DEBUG: print(self.comparisons-start)
        return node
 
    def heapify_down(self, parent_idx=0):
        ## GET CHILD NODE INDICES
        left_idx = 2*parent_idx + 1
        right_idx = 2*parent_idx + 2   
        swap_idx = parent_idx     

        ## FIND SMALLEST CHILD
        if left_idx < self.size and self.compare(swap_idx, left_idx):
            swap_idx = left_idx
        if right_idx < self.size and self.compare(swap_idx, right_idx):
            swap_idx = right_idx
        ## SWAP WITH THE SMALLEST CHILD AND HEAPIFY
        if parent_idx != swap_idx:
            self.swap(parent_idx, swap_idx)
            self.heapify_down(swap_idx)
        ## ELSE BASE CASE RETURN NONE
        return

    def heapify_up(self, child_idx):
        ## GET PARENT NODE INDEX
        parent_idx = (child_idx-1)//2 if child_idx % 2 else (child_idx-2)//2

        ## SWAP IF PARENT IS GREATER THAN CHILD
        if parent_idx >= 0 and self.compare(parent_idx, child_idx):
            self.swap(parent_idx, child_idx)
            child_idx = parent_idx
            self.heapify_up(child_idx)
        ## ELSE BASE CASE RETURN NONE
        return
    
    def compare(self, parent_idx, child_idx):
        self.comparisons += 1
        return self.array[parent_idx] > self.array[child_idx]

    def swap(self, parent_idx, child_idx):
        temp = self.array[parent_idx]
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
    # print(comparisons)
    return pairs, comparisons


## ANALYSYS
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def test_worst_case(n=100):
    cases = range(10, n+1, 10)
    comparisons = []
    m_cases = []
    for n in cases:
        p = utils.sample_p(n)
        m = utils.combinations(n)
        _, k = closest_pairs(p, m)
        comparisons.append(k)
        m_cases.append(m)

    fig = plt.figure(figsize=(4, 3), dpi=300)
    ax = fig.add_subplot(111)

    cases = np.array(cases)
    m_cases = np.array(m_cases)
    # program = ax.plot(cases, comparisons, color="orange", label="Program Time")
    program = ax.scatter(cases, comparisons, marker=".", linewidths=0.1, color="#66ABF7", label="Program Time")
    m_log_m = ax.plot(cases, m_cases*np.log2(m_cases), color="#F76666", label="M log M Time")
        
    fig.suptitle("Time Complexity Analysis", fontsize=10, fontweight='normal', alpha=0.5, y=0.96)
    ax.set_ylabel("Comparisons", fontweight="normal", alpha=0.5, fontsize="x-small")
    ax.set_xlabel("Number of Points", fontweight="normal", alpha=0.5, fontsize="x-small")
    ax.tick_params(axis='both', which='major', labelsize=3)
    ax.legend(loc=1, prop={'size': 4})

    fig.canvas.draw()
    fig.savefig(f"worst_case_{cases[-1]+1}.png")
    
    return
    

def test_values_of_m(n=100):
    p = utils.sample_p(n)
    cases = range(10,n+1,5)
    M_comparisons = []
    M_3_4_comparisons = []
    M_2_comparisons = []
    M_4_comparisons = []
    M_8_comparisons = []
    M_cases = []
    for n in cases:
        p = utils.sample_p(n)
        M = utils.combinations(n)
        M_cases.append(M)

        _, k = closest_pairs(p, M)
        M_comparisons.append(k)

        _, k = closest_pairs(p, 3*M//4)
        M_3_4_comparisons.append(k)

        _, k = closest_pairs(p, M//2)
        M_2_comparisons.append(k)

        _, k = closest_pairs(p, M//4)
        M_4_comparisons.append(k)

        _, k = closest_pairs(p, M//8)
        M_8_comparisons.append(k)

    fig = plt.figure(figsize=(4, 3), dpi=300)
    ax = fig.add_subplot(111)

    cases = np.array(cases)
    M_cases = np.array(M_cases)
    M_ = ax.scatter(cases, M_comparisons, marker=".", edgecolors=None, linewidths=0.001, color="#FFB637", label="M")
    M34 = ax.scatter(cases, M_3_4_comparisons, marker=".", edgecolors=None, linewidths=0.01, color="#FFDD26", label="0.75 M")
    M2 = ax.scatter(cases, M_2_comparisons, marker=".", edgecolors=None, linewidths=0.01, color="#9EF766", label="0.5 M")
    M4 = ax.scatter(cases, M_4_comparisons, marker=".", edgecolors=None, linewidths=0.01, color="#66ABF7", label="0.25 M")
    M8 = ax.scatter(cases, M_8_comparisons, marker=".", edgecolors=None, linewidths=0.01, color="#9A82ED", label="0.125 M")
    mlogm = ax.plot(cases, M_cases*np.log2(M_cases), color="#F76666", label="M log M Time")
    m2logm = ax.plot(cases, 2*M_cases*np.log2(M_cases), color="#2F2F2F", label="2M log M Time")

    fig.suptitle("Comparisons for M = bin(n, 2)", fontsize=10, fontweight='normal', alpha=0.5, y=0.96)
    ax.set_ylabel("Comparisons", fontweight="normal", alpha=0.5, fontsize="x-small")
    ax.set_xlabel("Number of Points", fontweight="normal", alpha=0.5, fontsize="x-small")
    ax.tick_params(axis='both', which='major', labelsize=3)
    ax.legend(loc=1, prop={'size': 4})


    fig.canvas.draw()
    fig.savefig(f"m_analysis.png")

    return
    
    TEAM_COLORS = {
        "blue": "#66ABF7",
        "red": "#F76666",
        "neutral": "#BDBDBD",
        "green": "#9EF766",
        "purple": "#9A82ED",
        "orange": "#FFB637",
        "yellow": "#FFDD26",
        "pink": "#FBAFF9",
        "black": "#2F2F2F",
    }




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
    p = utils.sample_p(n)
    # p = utils.sample_p_flat(n)
    # m = utils.sample_m(n)
    m = utils.combinations(n)
    # m = 10

    ans, comp = closest_pairs(p, m)
    t = [a.dist for a in ans]

    # test_worst_case(100)
    test_values_of_m(300)








