import os
import tensorflow as tf
import matplotlib.pyplot as plt

base_dir = './Cat_Vs_Dog/cats_and_dogs_images'
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')
test_dir = os.path.join(base_dir, 'test')

#Rescale all pixel values between 0 - 255 to (1, 0)
train_idg = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale = 1./255,
    rotation_range = 40,
    width_shift_range = 0.2,
    height_shift_range = 0.2,
    shear_range = 0.2,
    zoom_range = 0.2,
    horizontal_flip = True,
    fill_mode = 'nearest')
test_idg = tf.keras.preprocessing.image.ImageDataGenerator(rescale = 1./255) #Validation images are not augmented

train_gen = train_idg.flow_from_directory(
    train_dir,
    target_size = (150, 150),       #resize images to (150, 150)
    batch_size = 64,                #64 samples per batch i.e. shape = (64, 150, 150, 3)
    class_mode = 'binary'           #since this is a binary classification problem
)

valid_gen = test_idg.flow_from_directory(
    validation_dir,
    target_size = (150, 150),
    batch_size = 64,
    class_mode = 'binary')
    
vgg16_base = tf.keras.applications.VGG16(weights = 'imagenet',
                       include_top = False,
                      input_shape = (150, 150, 3))  # input_shape is an optional parameter
vgg16_base.summary()
model = tf.keras.Sequential()
model.add(vgg16_base)

model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(256, activation = 'relu')) 
model.add(tf.keras.layers.Dense(1, activation = 'sigmoid'))

model.summary()

tf.keras.utils.plot_model(model, show_shapes = True, to_file = 'pretrained_vgg16.png')

vgg16_base.trainable = False

model.compile(optimizer = tf.keras.optimizers.RMSprop(lr = 2e-5), # Try Adam
             loss = tf.keras.losses.binary_crossentropy,
             metrics = [tf.keras.metrics.binary_accuracy])

history = model.fit(
    train_gen,
    steps_per_epoch = 100,
    epochs = 30,
    validation_data = valid_gen,
    validation_steps = 50)


train_acc = history.history['binary_accuracy']
val_acc = history.history['val_binary_accuracy']
train_loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(train_acc) + 1)

plt.plot(epochs, train_acc, 'ro', label = 'Training Accuracy')
plt.plot(epochs, val_acc, 'g', label = 'Validation Accuracy')
plt.title('Accuracy Plot')
plt.legend()
plt.figure()
plt.plot(epochs, train_loss, 'go', label = 'Training Loss')
plt.plot(epochs, val_loss, 'r', label = 'Validation Loss')
plt.title('Loss Plot')
plt.legend()
plt.show()
