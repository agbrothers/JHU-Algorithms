ENVIRONMENT

This project was coded in python 3.8 and made use of the following libraries:
* numpy          -> for logarithms and array data structures
* matplotlib     -> for plotting
* tqdm           -> for visualizing loop progress
* argparse       -> for reading command line arguments
* math           -> for sqrt

FILES
* neighbors.py   -> data structures and core algorithm
* utils.py       -> helper functions for math and generating sample data
* plotting.py    -> functions for visualizing the algorithms performance on test data

RUNNING
To test this code, you will want to run the neighbors.py file. The following arguments are available:

-n ~ the number of input points
-m ~ the number of closest pairs to return
-s ~ the simplicity of the input data

-s=True will generate test data points along the x axis in the following form:

    `p = [(0,1), (0,4), ..., (0,n^2)]`

This type of input data is used to more easily verify the correctness of the algorithm. If the -s argument
is not passed in the command line, p samples n uniform random points between (-1,-1) and (1,1). 

For some reason, passing -s=False also generates simple data, so if you want random uniform data simply omit
the -s argument. 


EXAMPLES
From the root directory of this project, run:

python neighbors.py 

python neighbors.py -n=300 -m=1000

python neighbors.py -n=300 -s=True

python neighbors.py -n=10 -m=30 -s=True


PLOTTING 
To reproduce the plot in the report, from the root directory run:

python plotting.py


OPTIONAL
In neighbors.py there is a DEBUG=False flag. If set to True, a number of assert and
print statements will activate to ensure that mins are returned from the min heap, 
maxes are returned from the max heap, theoretical bounds on the numbers of expected
comparisons arent violated, etc. Enabling this flag will significantly hamper the 
performance of the program.
