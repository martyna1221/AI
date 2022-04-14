from scipy.linalg import eigh
import numpy as np
import matplotlib.pyplot as plt


def load_and_center_dataset(filename):
    x = np.load(filename) # loads file data
    mean = np.mean(x, axis = 0) # calculates mean of loaded data
    data = x - mean  # centers the loaded data
    return np.array(data, dtype = float) # returns the centered data as a NumPy array of floats

def get_covariance(dataset):
    n = len(dataset) # gets n (the no. of images)
    dot_p = np.dot(np.transpose(dataset), dataset) # calculates the dot product of x^T and x
    return np.array((dot_p) * (1 / (n - 1))) # returns the calculated covariance matrix

def get_eig(S, m):
    d = len(S) # gets the number of rows/cols. of matrix S

    # gets largest m e-vals and corr. e-vectors
    e_val, e_vector = eigh(a = S, eigvals = (d - m, d - 1))
    return np.array(np.diag(sorted(e_val, reverse = True))), np.array(np.flip(e_vector, axis = 1))

def get_eig_perc(S, perc):
    arr = [] # creates an array for e-vals above a certain percent
    sum = 0 

    e_val, e_vector = eigh(a = S) # gets e-vals and corr. e-vectors

    for i in e_val: # iterates over all e-vals and adds them together
        sum += i
    for k in e_val: # iterates over all e-vals and sees if they are above a certain percent
        if ((k / sum) > perc):
            arr.append(k) # if above a certain percent, these e-vals are added to an array

    d = len(S) # gets the number of rows/cols. of matrix S
    m = len(arr) # gets the number of elements in array arr

    # gets largest m e-vals and corr. e-vectors that are above threshold perc
    e_val, e_vector = eigh(a = S, eigvals = (d - m, d - 1)) 
    return np.array(np.diag(sorted(e_val, reverse = True))), np.array(np.flip(e_vector, axis = 1))

def project_image(img, U):
    # gets the dot product of U^T (a matrix of e-vectors) and an image from the dataset
    a = np.dot(np.transpose(U), img) 

    # returns the dot product of U (a matrix of e-vectors) and the above computed alphas of an image 
    return np.array(np.dot(U, a))

def display_image(orig, proj):
    orig = np.reshape(orig, (32, 32)) # reshapes orig to be a 32x32 matrix
    proj = np.reshape(proj, (32, 32)) # reshapes proj to be a 32x32 matrix

    orig = np.rot90(orig, k = 3) # rotates orig k * 90 deg. = 270 deg.
    proj = np.rot90(proj, k = 3) # rotates proj k * 90 deg. = 270 deg.

    # creates 2 subplots & resizes them for color bar alignment
    fig, (p1, p2) = plt.subplots(1, 2, figsize = (13, 3)) 

    orig_img = p1.imshow(orig, aspect = 'equal') # renders orig
    p1.set_title('Original') # titles the subplot p1 'Original'
    fig.colorbar(orig_img, ax = p1) # creates a colorbar for subplot p1

    proj_img = p2.imshow(proj, aspect = 'equal') # renders proj
    p2.set_title('Projection') # titles the subplot p2 'Projection'
    fig.colorbar(proj_img, ax = p2) # creates a colorbar for subplot p2

    return plt.show() # returns the rendered plots