import tensorflow as tf
import os

def _dsep_conv(inp, convs):
    x = inp
    for conv in convs:
        if conv['stride'] > 1 : x = tf.keras.layers.ZeroPadding2D(((1,0), (1,0)))(x)
        x = tf.keras.layers.SeparableConv2D(conv['filter'],
                                           conv['kernel'],
                                           strides = conv['stride'],
                                           padding = 'valid' if conv['stride'] > 1 else 'same',
                                           name = 'dsep_conv_' + str(conv['layer_idx']),
                                           use_bias = False if conv['bnorm'] else True,
                                           activation = 'relu' if conv['relu'] else None)(x)
        if conv['bnorm']: x = tf.keras.layers.BatchNormalization(
            epsilon = 0.001,
            name = 'bnorm_' + str(conv['layer_idx']))(x)
        if conv['leaky']: x = tf.keras.layers.LeakyReLU(alpha = 0.1, 
                                                        name = 'leaky_' + str(conv['layer_idx']))(x)
    return x


def make_dsep_conv_model():
   input_image = tf.keras.layers.Input(shape = (None, None, 3))
   x = _dsep_conv(input_image, [{'filter': 32, 'kernel': 3, 'stride': 1, 'bnorm': True, 'relu': True, 'leaky': False, 'layer_idx': 0}])
   x = tf.keras.layers.MaxPooling2D(2)(x)

   x = _dsep_conv(x, [{'filter': 64, 'kernel': 3, 'stride': 1, 'bnorm': True, 'relu': True, 'leaky': False, 'layer_idx': 2}])
   x = tf.keras.layers.MaxPooling2D(2)(x)

   x = _dsep_conv(x, [{'filter': 128, 'kernel': 3, 'stride': 1, 'bnorm': True, 'relu': True, 'leaky': False, 'layer_idx': 3}])
   x = tf.keras.layers.MaxPooling2D(2)(x)
   x = _dsep_conv(x, [{'filter': 128, 'kernel': 3, 'stride': 1, 'bnorm': True, 'relu': True, 'leaky': False, 'layer_idx': 4}])
   x = tf.keras.layers.GlobalAveragePooling2D()(x)

   x = tf.keras.layers.Flatten()(x)
   x = tf.keras.layers.Dropout(0.5)(x)
   x = tf.keras.layers.Dense(512, activation = 'relu')(x)
   output = tf.keras.layers.Dense(1, activation = 'sigmoid')(x)
    
   model = tf.keras.models.Model(input_image, output, name = 'cat-dog_dsep_conv_model')
   return model
    
model = make_dsep_conv_model()
model.compile(optimizer = tf.keras.optimizers.RMSprop(lr = 1e-4),
             loss = tf.keras.losses.binary_crossentropy,
             metrics = [tf.keras.metrics.binary_accuracy])


base_dir = './cats_and_dogs_images'
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
    batch_size = 32,                #32 samples per batch i.e. shape = (32, 150, 150, 3)
    class_mode = 'binary'           #since this is a binary classification problem
)

valid_gen = test_idg.flow_from_directory(
    validation_dir,
    target_size = (150, 150),
    batch_size = 32,
    class_mode = 'binary')

history = model.fit(
    train_gen,
    steps_per_epoch = 100,
    epochs = 100,
    validation_data = valid_gen,
    validation_steps = 50)
