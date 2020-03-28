import numpy as np

def add_gaussian_noise(original_image, sigma):
    gaussian_noise = np.random.normal(0, sigma, np.shape(original_image))
    noisy_img = original_image + gaussian_noise
    noisy_img[noisy_img < 0] = 0
    noisy_img[noisy_img > 255] = 255
    return noisy_img
