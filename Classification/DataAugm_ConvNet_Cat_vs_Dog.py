import os
import tensorflow as tf

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
    batch_size = 32,                #32samples per batch i.e. shape = (32, 150, 150, 3)
    class_mode = 'binary'           #since this is a binary classification problem
)

valid_gen = test_idg.flow_from_directory(
    validation_dir,
    target_size = (150, 150),
    batch_size = 32,
    class_mode = 'binary')
    
model = tf.keras.Sequential()
model.add(tf.keras.layers.Conv2D(filters = 32, kernel_size = 3, activation = 'relu',
                input_shape = (150, 150, 3))) #channel last; num of params = (32 * ((3 * 3 * 3) + 1)) = 896
model.add(tf.keras.layers.MaxPooling2D(pool_size = (2, 2)))

model.add(tf.keras.layers.Conv2D(filters = 64, kernel_size = 3, activation = 'relu'))
model.add(tf.keras.layers.MaxPooling2D(pool_size = (2, 2)))

model.add(tf.keras.layers.Conv2D(filters = 128, kernel_size = 3, activation = 'relu'))
model.add(tf.keras.layers.MaxPooling2D(pool_size = (2, 2)))

model.add(tf.keras.layers.Conv2D(filters = 128, kernel_size = 3, activation = 'relu'))
model.add(tf.keras.layers.MaxPooling2D(pool_size = (2, 2)))

model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(512, activation = 'relu'))
model.add(tf.keras.layers.Dense(1, activation = 'sigmoid'))

model.summary()

tf.keras.utils.plot_model(model, show_shapes = True, to_file = 'data-aug_convnet.png')

model.compile(optimizer = tf.keras.optimizers.RMSprop(lr = 1e-4),
             loss = tf.keras.losses.binary_crossentropy,
             metrics = [tf.keras.metrics.binary_accuracy])

callbacks_list = [
    tf.keras.callbacks.EarlyStopping(
        monitor = 'binary_accuracy'
        , patience = 10
    ,),
    tf.keras.callbacks.ModelCheckpoint(
        filepath = 'cat_vs_dog_2.h5'
        , monitor = 'val_loss'
        , save_best_only = True
    ,),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor = 'val_loss'
        , factor = 0.1
        , patience = 10
        ,
    )
]

history = model.fit(
    train_gen,
    steps_per_epoch = 100,
    epochs = 100,
    callbacks = callbacks_list,  #Comment out this line if you want to let all the epochs run
    validation_data = valid_gen,
    validation_steps = 50)

model.save('cat_vs_dog_with_data_aug.h5')

def smoothening(points, factor = 0.8):
    smooth_points = []
    for point in points:
        if smooth_points:
            previous = smooth_points[-1]
            smooth_points.append(previous * factor + point * (1 - factor))
        else:
            smooth_points.append(point)
    return smooth_points
 
 import matplotlib.pyplot as plt
train_acc = history.history['binary_accuracy']
val_acc = history.history['val_binary_accuracy']
train_loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(train_acc) + 1)

plt.plot(epochs, smoothening(train_acc), 'go', label = 'Training Accuracy')
plt.plot(epochs, smoothening(val_acc), 'r', label = 'Validation Accuracy')
plt.title('Accuracy Plot')
plt.legend()
plt.figure()
plt.plot(epochs, smoothening(train_loss), 'ro', label = 'Training Loss')
plt.plot(epochs, smoothening(val_loss), 'g', label = 'Validation Loss')
plt.title('Loss Plot')
plt.legend()
plt.show()
