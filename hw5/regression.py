import numpy as np
from matplotlib import pyplot as plt
import csv
import math
import random

# Feel free to import other packages, if needed.
# As long as they are supported by CSL machines.


def get_dataset(filename):
    dataset = None
    l = [] # list used to append rows from the csv file

    file = open(filename) # opens a specified file
    reader = csv.reader(file) # reads from said file
    next(reader) # skips over first row (i.e. the row of labels)

    for row in reader: # iterates over every row in said file
        del row[0] # deletes entry 0 of each row (i.e. IDNO entry)
        l.append(row) # adds each cleaned row to list 'l'

    dataset = np.array(l) # makes the list 'l' into a numpy array
    dataset = dataset.astype(float) # converts all entries to be float type

    file.close() # closes opened file named 'file'

    return dataset # returns a cleaned dataset


def print_stats(dataset, col):
    sum_of_all = 0 # used as var. to sum all elements of a given col. (for mean calculation)
    sd_difference = 0 # used as var. to sum all squared diff. of a given col. (for sd calculation)
    transpose = dataset.transpose() # gets the transpose of a given matrix (easier for later calculation)

    num_elements = np.shape(transpose[col])[0] # gets the # of elements in a given col.

    print(num_elements) # prints the # of elements in a given col.

    for element in transpose[col]: # iterates through all elements of a col. & adds them all together
        sum_of_all += element
    
    mean = sum_of_all / num_elements # calculates mean of a given col.
    formatted_mean = "{0:.2f}".format(mean) # formats mean to have exactly 2 decimal points

    print(formatted_mean) # prints the formatted mean to the console

    for element in transpose[col]: # iterates through all elements of a col. & adds all squared diff. together
        sd_difference += ((element - mean) ** 2)

    sd = math.sqrt(sd_difference / (num_elements - 1)) # calculates standard deviation
    formatted_sd = "{0:.2f}".format(sd) # formats sd to have exactly 2 decimal points

    print(formatted_sd) # prints the formatted sd to console

    pass


def regression(dataset, cols, betas):
    mse = None
    total_sum = 0 # used as var. to add all square sums of f(x) 

    for element in dataset: # iterates through all 252 elements in the dataset 
        intermediate = 0 # used as var. to add all compoments of f(x)
        intermediate += betas[0] # adds the first beta to the var. 'intermediate'
        i = 0
        while i < len(cols): # while loop that iterates through all 'cols' elements
            intermediate += element[cols[i]] * betas[i + 1] # adds all intermediate components to 'intermediate'
            i += 1 # increments the while loop
        intermediate -= element[0] # subtracts y from 'intermediate' (i.e. bodyfat)
        intermediate = intermediate ** 2 # squares intermediate
        total_sum += intermediate # adds the found component to 'total_sum' for easier access later on

    mse = (total_sum / np.shape(dataset)[0]) # calculates MSE by dividing the sum of all component by the numb. of rows

    return mse


def gradient_descent(dataset, cols, betas):
    grads = None
    total_sum = 0 # used as var. to add all sums of f(x) (dependent on beta)
    partial_d = [] # used as var. to store all computed partial derivatives (dependent on beta)

    i = 0
    while i < len(betas): # iterates over all betas (easier later calculation)
        if i == 0: # for beta_0 ...
            for element in dataset: # iterates over all elements in dataset (similar to method above)
                intermediate = 0
                intermediate += betas[0]
                k = 0
                while k < len(cols):
                    intermediate += element[cols[k]] * betas[k + 1]
                    k += 1
                intermediate -= element[0]
                total_sum += intermediate
            partial_d_b0 = (2 / np.shape(dataset)[0]) * total_sum # calculates beta_0 partial derivative
            total_sum = 0 # zeros the total_sum for next calculation
            partial_d.append(partial_d_b0) # adds partial derivative of beta_0 to list 'partial_d'
        else: # for beta != beta_0 ...
            for element in dataset: # iterates over all elements in dataset (similar to method above)
                intermediate = 0
                intermediate += betas[0]
                k = 0
                while k < len(cols):
                    intermediate += element[cols[k]] * betas[k + 1]
                    k += 1
                intermediate -= element[0]
                intermediate = intermediate * element[cols[i - 1]]
                total_sum += intermediate
            partial_d_bx = (2 / np.shape(dataset)[0]) * total_sum # calculates beta_x partial derivative
            total_sum = 0 # zeros the total_sum for next calculation
            partial_d.append(partial_d_bx) # adds partial derivative of beta_x to list 'partial_d'
        i += 1 # increments the while loop

    grads = np.array(partial_d) # converts list 'partial_d' to a numpy array

    return grads # add additional ouput if necessary (i.e. like the writeup)


def iterate_gradient(dataset, cols, betas, T, eta):
    i = 0
    while i < T: # iterates the necessary loops T times 
        k = 0
        beta_list = []
        print_beta_list = []
    
        while k < len(betas): # updates k number of betas for a specified input
            derivative = gradient_descent(dataset, cols, betas)
            beta = betas[k] - (eta * derivative[k])
            print_beta_list.append("{0:.2f}".format(beta))
            beta_list.append(beta)
            k += 1
        
        # prints out properly formatted T, MSE, beta_0, beta_1, ..., beta_m at each iteration T 
        print(i + 1, "{0:.2f}".format(regression(dataset, cols, beta_list)), *print_beta_list, sep = " ")
        betas = beta_list # updates 'betas' with the newly claculated beta_list
        beta_list = [] # clears 'beta_list'
        print_beta_list = [] # clears 'print_beta_list'
        i += 1

    pass


def compute_betas(dataset, cols):
    betas = None
    mse = None

    y = dataset[:, 0] # gets col. 0 of dataset (i.e. bodyfat)
    X = dataset[:, cols] # gets the cols. that correspond to the 'cols' parameter
    
    X = np.hstack((np.ones((np.shape(X)[0], 1)), X)) # appends a col. of 0's to beginning array X

    inverse = np.linalg.inv(np.dot(np.transpose(X), X)) # (X^T * X)^-1
    p1 = np.dot(inverse, np.transpose(X)) # (X^T * X)^-1 * X^T
    betas = np.dot(p1, y) # (X^T * X)^-1 * X^T * y
    mse = regression(dataset, cols, betas) # calculates MSE with new betas

    return (mse, *betas)


def predict(dataset, cols, features):
    result = None

    _, *betas = compute_betas(dataset, cols) # gets the corresponding closed-form betas
    prediction = 0 # var. used for summation of all parts of f(x)

    b = 0
    while b < len(betas): # iterates through the elements in 'betas'
        if b == 0: # when the 'betas' element == b_0
            prediction += betas[b]
        else: # when the 'betas' element == b_x
            prediction += betas[b] * features[b - 1]
        b += 1 # increments the while loop
    result = prediction

    return result # returns the predicted bodyfat percentage


def synthetic_datasets(betas, alphas, X, sigma):
    n = np.shape(X)[0] # gets the number of rows of X

    l = np.zeros((n, 2)) # linear model

    i = 0
    while i < n: # puts associated x values in the 2nd col. of the new array
        l[i][1] = X[i]
        i += 1

    r = 0
    while r < n: # linear model calculation for each x
        y = betas[0] + (betas[1] * l[r][1]) + np.random.normal(0, sigma)
        l[r][0] = y
        r += 1
    
    q = np.zeros((n, 2)) # quadratic model
    
    k = 0
    while k < n: # puts associated x values in the 2nd col. of the new array
        q[k][1] = X[k]
        k += 1
    
    r = 0
    while r < n: # quadratic model calculation for each x
        y = alphas[0] + (alphas[1] * (q[r][1] ** 2)) + np.random.normal(0, sigma)
        q[r][0] = y
        r += 1

    return l, q


def plot_mse():
    from sys import argv
    if len(argv) == 2 and argv[1] == 'csl':
        import matplotlib
        matplotlib.use('Agg')

    n = 1000
    X = np.zeros((n, 1)) # creates an [1000, 1] vector

    i = 0
    while i < n: # randomizes 1000 inputs for vecor X
        X[i][0] = random.randint(-100, 100)
        i += 1
    
    betas = [1, 2] # random betas
    alphas = [3, 4] # random alphas
    sigma = [.0001, .001, .01, .1, 1, 10, 100, 1000, 10000, 100000] # sigma list

    m = [] # var. that will be used to hold linear MSEs
    n = [] # var. that will be used to hold quadratic MSEs

    # 20 synthetic datasets (10 linear & 10 quadratic) were created with various sigma values
    l1, q1 = synthetic_datasets(betas, alphas, X, sigma[0])
    msel1, *_ = compute_betas(l1, cols = [1]) # linear MSE calcluation
    mseq1, *_ = compute_betas(q1, cols = [1]) # quadratic MSE calcluation
    m.append(msel1)
    n.append(mseq1)

    l2, q2 = synthetic_datasets(betas, alphas, X, sigma[1])
    msel2, *_ = compute_betas(l2, cols = [1])
    mseq2, *_ = compute_betas(q2, cols = [1])
    m.append(msel2)
    n.append(mseq2)

    l3, q3 = synthetic_datasets(betas, alphas, X, sigma[2])
    msel3, *_ = compute_betas(l3, cols = [1])
    mseq3, *_ = compute_betas(q3, cols = [1])
    m.append(msel3)
    n.append(mseq3)

    l4, q4 = synthetic_datasets(betas, alphas, X, sigma[3])
    msel4, *_ = compute_betas(l4, cols = [1])
    mseq4, *_ = compute_betas(q4, cols = [1])
    m.append(msel4)
    n.append(mseq4)

    l5, q5 = synthetic_datasets(betas, alphas, X, sigma[4])
    msel5, *_ = compute_betas(l5, cols = [1])
    mseq5, *_ = compute_betas(q5, cols = [1])
    m.append(msel5)
    n.append(mseq5)

    l6, q6 = synthetic_datasets(betas, alphas, X, sigma[5])
    msel6, *_ = compute_betas(l6, cols = [1])
    mseq6, *_ = compute_betas(q6, cols = [1])
    m.append(msel6)
    n.append(mseq6)

    l7, q7 = synthetic_datasets(betas, alphas, X, sigma[6])
    msel7, *_ = compute_betas(l7, cols = [1])
    mseq7, *_ = compute_betas(q7, cols = [1])
    m.append(msel7)
    n.append(mseq7)

    l8, q8 = synthetic_datasets(betas, alphas, X, sigma[7])
    msel8, *_ = compute_betas(l8, cols = [1])
    mseq8, *_ = compute_betas(q8, cols = [1])
    m.append(msel8)
    n.append(mseq8)

    l9, q9 = synthetic_datasets(betas, alphas, X, sigma[8])
    msel9, *_ = compute_betas(l9, cols = [1])
    mseq9, *_ = compute_betas(q9, cols = [1])
    m.append(msel9)
    n.append(mseq9)

    l10, q10 = synthetic_datasets(betas, alphas, X, sigma[9])
    msel10, *_ = compute_betas(l10, cols = [1])
    mseq10, *_ = compute_betas(q10, cols = [1])
    m.append(msel10)
    n.append(mseq10)

    fig = plt.figure()
    plt.xlabel("Sigma") # x-axis label
    plt.ylabel("MSEs") # y-axis label
    plt.plot(sigma, m, marker = "o", label = "Linear MSEs") # linear MSEs plotted w/ specified features
    plt.plot(sigma, n, marker = "o", label = "Quadratic MSEs") # quadratic MSEs plotted w/ specified features
    plt.xscale("log") # x-axis scale
    plt.yscale("log") # y-axis scale
    plt.legend(loc = 4) # plots legend in lower-right corner of graph

    plt.savefig('./mse.pdf') # saves output as a file name 'mse.pdf'

if __name__ == '__main__':
    ### DO NOT CHANGE THIS SECTION ###
    plot_mse()
