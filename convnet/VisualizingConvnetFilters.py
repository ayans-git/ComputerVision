#Reference: Deep Learning with Python by Francois Chollet
from keras.applications import VGG16
from keras import backend as K
import numpy as np


#STEP: Utility function to convert a tensor into a valid image
def deprocess_image(x):
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1   #normalizes the tensor: centers on 0, ensures that std is 0.1
    
    x += 0.5
    x = np.clip(x, 0, 1)  #clips to [0, 1]
    
    #Converts to an RGB array
    x *= 255
    x = np.clip(x, 0, 255).astype('uint8')
    return x


#STEP: Function to generate filter visualization
def generate_pattern(layer_name, filter_index, height = 150, width = 150):
    #Builds a loss function that maximizes the activation
    #of the nth filter of the layer under consideration
    
    model = VGG16(weights = 'imagenet',
             include_top = False
             )
    layer_output = model.get_layer(layer_name).output
    loss = K.mean(layer_output[:, :, :, filter_index])
    
    #Computes the gradient of the input picture with regard to this loss
    grads = K.gradients(loss, model.input)[0]
    
    #Normalizes the gradient
    #ensuring no accidental division by 0
    grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)
    
    #Returns the loss and grads given the input picture
    iterate = K.function([model.input], [loss, grads])
    
    #Starts from a gray image with some noise
    input_img_data = np.random.random((1, height, width, 3)) * 20 + 128
    
    #Runs gradient ascent for 40 steps
    step = 1 #magnitude of each gradient update
    for iCount in range(40):
        loss_value, grads_value = iterate([input_img_data])
        input_img_data += grads_value * step
        
    img = input_img_data[0]
    return deprocess_image(img)


import matplotlib.pyplot as plt
plt.imshow(generate_pattern('block3_conv1', 0))


#STEP: Generating a grid of all filter response patterns in a layer
import numpy as np
layer_name = 'block3_conv1'
size = 64
margin = 5

results = np.zeros((8 * size + 7 * margin, 8 * size + 7 * margin, 3))

for irow in range(8):
    for jcol in range(8):
        #Generates the pattern for filter irow + (jcol * 8) in layer_name
        filter_img = generate_pattern(layer_name, irow + (jcol * 8), height = size, width = size)
        
        horizontal_start = irow * size + irow * margin
        horizontal_end = horizontal_start + size
        vertical_start = jcol * size + jcol * margin
        vertical_end = vertical_start + size
        results[horizontal_start : horizontal_end, 
               vertical_start : vertical_end, :] = filter_img
plt.figure(figsize = (20, 20))
plt.imshow(results)
