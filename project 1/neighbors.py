import numpy as np         ## for np.log2
from tqdm import trange    ## for displaying loop progress

## LOCAL PACKAGES
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
    
    ## DISPLAY CONTENTS OF NODE WHEN PRINTING
    def __repr__(self):
        return f"node: ({self.dist}, {self.p1}, {self.p2})"
    
    ## DISPLAY CONTENTS OF NODE WHEN PRINTING
    def __str__(self):
        return f"({self.dist}, {self.p1}, {self.p2})"


## GENERIC HEAP
class Heap:

    ## ARRAY BASED IMPLEMENTATION
    def __init__(self, m) -> None:
        self.array = []
        self.comparisons = 0
        self.max_depth = int(np.log2(m))

    def insert(self, node, idx=None):
        if DEBUG: start = self.comparisons
        ## ADD NEW NODE TO BOTTOM OF TREE
        if idx is None:
            self.array.append(node)
        ## INSERT NEW NODE IN A SPECIFIC INDEX
        else:
            self.array[idx] = node
        ## HEAPIFY
        self.heapify_up(idx or self.size-1)
        if DEBUG: print(self.comparisons-start)
        if DEBUG: assert self.depth <= self.max_depth

    def get_root(self):
        if DEBUG: start = self.comparisons
        if self.empty(): raise Exception("ERROR: Can't get min from an empty heap!")
        ## RETURN ROOT NODE
        node = self.array[0]
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

## MAX HEAP
class MaxHeap(Heap):

    def compare(self, parent_idx, child_idx):
        ## TRUE IF PARENT IS LESS THAN CHILD (swap required)
        self.comparisons += 1
        return self.array[parent_idx] < self.array[child_idx]

    def get_root(self):
        if DEBUG: mx = max(self.array)
        max_node = super().get_root()
        if DEBUG: assert max_node == mx
        return max_node

## MIN HEAP
class MinHeap(Heap):

    def compare(self, parent_idx, child_idx):
        ## TRUE IF PARENT IS GREATER THAN CHILD (swap required)
        self.comparisons += 1
        return self.array[parent_idx] > self.array[child_idx]

    def insert(self, node, idx=None):
        ## UPDATE THE NODE OBJECT IDX WHEN INSERTING
        node.idx = self.size if idx is None else idx
        super().insert(node, idx)

    def swap(self, parent_idx, child_idx):
        super().swap(parent_idx, child_idx)

        ## UPDATE THE NODE OBJECT IDX WHEN SWAPPING
        self.array[parent_idx].idx = parent_idx
        self.array[child_idx].idx = child_idx

    def get_root(self):
        if DEBUG: mn = min(self.array)
        min_node = super().get_root()
        if DEBUG: assert min_node == mn
        return min_node


## ALGORITHM

def closest_pairs(p, m):
    ## INIT HEAP
    min_heap = MinHeap(m)
    max_heap = MaxHeap(m)

    comparisons = 0

    ## CHECK m
    n = len(p)
    assert m <= utils.combinations(n), "ERROR: m is greater than the number of possible unique pairs of points"

    ## COMPUTE DIST BETWEEN EACH COMBINATION OF POINTS
    ## T(n) = \sum(n-1) -> M lg M  operation per step
    for i in trange(n):  # O(M log2 M) -> O(M log2 m) 

        for j in range(i+1,n):

            ## GET DIST
            dist = utils.distance(p[i], p[j])
            ## TRACK COMPARISONS
            comparisons += 1
            ## CONSTRUCT NODE
            node = Node(dist, p[i], p[j])
            ## INSERT NODE INTO HEAP

            min_start = min_heap.comparisons
            max_start = max_heap.comparisons

            ## WHEN > m ITEMS IN THE MIN HEAP, USE THE MAX HEAP 
            ## TO GET THE INDEX OF THE MIN HEAP'S LARGEST NODE 
            ## AND REPLACE IT WITH THE NEW NODE
            insert_idx = None
            if min_heap.size >= m:
                max_node = max_heap.get_root()
                insert_idx = max_node.idx
                assert max_heap.comparisons - max_start < 2*np.log2(m)
                max_start = max_heap.comparisons

            ## MAY WANT TO INSERT THE NEW NODE INTO THE MAX HEAP AFTERWARDS
            ## BOTH HEAPS CONTAIN ALL OF THE SAME NODES
            max_heap.insert(node) 
            min_heap.insert(node, insert_idx) 
            assert min_heap.comparisons - min_start < np.log2(m)
            assert max_heap.comparisons - max_start < np.log2(m)

    ## GET THE M CLOSEST PAIRS FROM THE HEAP
    pairs = []
    ## T(n) = m -> log m operations per step
    for i in trange(m): # O(m log2 M) -> O(m log2 m)
        node = min_heap.get_root()  ## log2 m
        # pairs.append((node.p1, node.p2))
        pairs.append(node)

    comparisons += min_heap.comparisons
    comparisons += max_heap.comparisons
    if DEBUG: print(comparisons)
    return pairs, comparisons


def main(args):

    n = args.n
    m = args.m
    simple = args.simple
    
    if simple == True:
        p = utils.sample_p_flat(n)
    else:
        p = utils.sample_p(n)

    M = utils.combinations(n)
    if m is None: m = utils.sample_m(n)

    print(f"\nRETURN THE {m} CLOSEST PAIRS OF THE FOLLOWING {n} POINTS ({M} possible pairs):\n")
    print(p)
    print()

    ans, comp = closest_pairs(p, m)

    print(f"\nPERFORMED {comp} COMPARISONS\n")
    print("SORTED CLOSEST PAIRS: (dist, point_1, point_2)")
    for node in ans:
        print(node)



if __name__ == "__main__":

    ### Library for adding filepaths as command line arguments
    import argparse
    parser = argparse.ArgumentParser(description="Encode/Decode using Huffman Coding")
    parser.add_argument('--n', '-n', type=int, default=100,
                        help='Number of random (x,y) points to generate.')
    parser.add_argument('--m', '-m', type=int, default=None,
                        help='Number of closest pairs of points to return, randomly sampled by default')
    parser.add_argument('--simple', '-s', type=bool, default=False,
                        help='Make input points very simple for debugging.')
    args,_ = parser.parse_known_args()
    ### 

    main(args)
