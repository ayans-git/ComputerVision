#Reference: Deep Learning with Python by Francois Chollet
#STEP: Load data
from keras.datasets import mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
(train_images_dense, train_labels_dense), (test_images_dense, test_labels_dense) = mnist.load_data()

#STEP: Build Dense Layered Model
from keras import models
from keras import layers
def build_Dense_Layer_Model():
    model = models.Sequential()
    model.add(layers.Dense(512, activation = 'relu', input_shape = (28 * 28,)))
    model.add(layers.Dense(10, activation = 'softmax'))
    return model
    
#STEP: Build CONVD Layered Model
def build_Convd_Layer_Model():
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation = 'relu', input_shape = (28, 28, 1)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation = 'relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation = 'relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation = 'relu'))
    model.add(layers.Dense(10, activation = 'softmax'))
    return model


#STEP: Compile Model
from keras import optimizers
from keras import losses
from keras import metrics
def compile_model(model):
    model.compile(optimizer = optimizers.RMSprop(lr = 0.001),
                 loss = losses.categorical_crossentropy,
                 metrics = [metrics.categorical_accuracy])
    return model
    

#STEP: Reshape data and labels
from keras.utils import to_categorical
train_images_dense = train_images_dense.reshape((60000, 28 * 28))
train_images_dense = train_images_dense.astype('float32') / 255

test_images_dense = test_images_dense.reshape((10000, 28 * 28))
test_images_dense = test_images_dense.astype('float32') / 255

train_labels_dense = to_categorical(train_labels_dense)
test_labels_dense = to_categorical(test_labels_dense)

train_images = train_images.reshape((60000, 28, 28, 1))
train_images = train_images.astype('float32') / 255

test_images = test_images.reshape((10000, 28, 28, 1))
test_images = test_images.astype('float32') / 255

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)


#STEP: Compile and Train Models
dense_model = build_Dense_Layer_Model()
dense_model = compile_model(dense_model)
history_dense = dense_model.fit(train_images_dense 
                                ,train_labels_dense 
                                ,epochs = 5 
                                ,batch_size = 64
                                #,verbose = 0
                               )

conv_model = build_Convd_Layer_Model()
conv_model = compile_model(conv_model)
history_conv = conv_model.fit(train_images
                             ,train_labels
                             ,epochs = 5
                             ,batch_size = 64
                             #,verbose = 0
                             )
                             

conv_model.summary()

#STEP: For both models plot accuracy
import matplotlib.pyplot as plt
accuracy_dl = history_dense.history['categorical_accuracy']
accuracy_cl = history_conv.history['categorical_accuracy']

epochs = range(1, len(accuracy_dl) + 1)

plt.plot(epochs, accuracy_dl, 'bo', label = 'Dense Layer Accuracy')
plt.plot(epochs, accuracy_cl, 'r+', label = 'Conv Layer Accuracy')
plt.title('Accuracy Comparison')
plt.xlabel('Epochs')
plt.ylabel('Training Accuracy')
plt.legend()
plt.show()

