import numpy as np
from scipy.signal import gaussian

def create_gaussian_kernel(kernel_size = 3):
    kernel = gaussian(kernel_size, kernel_size/3).reshape(kernel_size, 1)
    kernel = np.dot(kernel, kernel.transpose())
    kernel /= np.sum(kernel)
    return kernel
