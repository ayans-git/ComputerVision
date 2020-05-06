import numpy as np

def median_filter(noisy_image, kernel_size):
    tmp_image = []
    index = kernel_size // 2
    filtered_image = []
    filtered_image = np.zeros((len(noisy_image),len(noisy_image[0])))
    for iCount in range(len(noisy_image)):
        for jCount in range(len(noisy_image[0])):
            for zCount in range(kernel_size):
                if iCount + zCount - index < 0 or iCount + zCount - index > len(noisy_image) - 1:
                    for c in range(kernel_size):
                        tmp_image.append(0)
                else:
                    if jCount + zCount - index < 0 or jCount + index > len(noisy_image[0]) - 1:
                        tmp_image.append(0)
                    else:
                        for k in range(kernel_size):
                            tmp_image.append(noisy_image[iCount + zCount - index][jCount + k - index])
            tmp_image.sort()
            filtered_image[iCount][jCount] = tmp_image[len(tmp_image) // 2]
            tmp_image = []
    return filtered_image
