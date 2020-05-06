import numpy as np
from numpy.fft import fft2, ifft2

def wiener_filter(noisy_image, kernel, K):
    kernel /= np.sum(kernel)
    filtered_image = np.copy(noisy_image)
    filtered_image = fft2(filtered_image)
    kernel = fft2(kernel, s = noisy_image.shape)
    kernel = np.conj(kernel) / (np.abs(kernel) ** 2 + K)
    filtered_image = filtered_image * kernel
    filtered_image = np.abs(ifft2(filtered_image))
    return filtered_image
