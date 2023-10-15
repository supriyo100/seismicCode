#CBAM-UNet separetely
# Import functions
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D, concatenate, Lambda, multiply, add, GlobalAveragePooling2D, Reshape, Dense
from tensorflow.keras import backend as K


# Define a function for the Channel Attention Module (CAM), which will take the output of a convolutional
# layer as input and output the features 

def channel_attention_module(input_feature, ratio=8):
    channel = input_feature.shape[-1]
    
    shared_layer_one = Dense(channel//ratio, activation='relu', kernel_initializer='he_normal', use_bias=True, bias_initializer='zeros')
    shared_layer_two = Dense(channel, activation='relu', kernel_initializer='he_normal', use_bias=True, bias_initializer='zeros')

    avg_pool = GlobalAveragePooling2D()(input_feature)
    avg_pool = Reshape((1, 1, channel))(avg_pool)
    avg_pool = shared_layer_one(avg_pool)
    avg_pool = shared_layer_two(avg_pool)

    max_pool = tf.reduce_max(input_feature, axis=[1, 2], keepdims=True)
    max_pool = shared_layer_one(max_pool)
    max_pool = shared_layer_two(max_pool)

    feature = add([avg_pool, max_pool])
    feature = tf.nn.sigmoid(feature)
    return multiply([input_feature, feature])
def spatial_attention_module(input_feature):
    kernel_size = 7
    max_pool = tf.reduce_max(input_feature, axis=3, keepdims=True)
    avg_pool = tf.reduce_mean(input_feature, axis=3, keepdims=True)

    concat = tf.concat([max_pool, avg_pool], 3)
    concat = Conv2D(1, kernel_size, padding='same', activation='sigmoid', kernel_initializer='he_normal')(concat)

    return multiply([input_feature, concat])

inputs = tf.keras.layers.Input((128,128,1))

#Contraction path
c1 = tf.keras.layers.Conv2D(16, (3, 3), activation='tanh', kernel_initializer='he_normal', padding='same')(inputs)
c1 = channel_attention_module(c1)
c1 = spatial_attention_module(c1)
c1 = tf.keras.layers.Dropout(0.1)(c1)
c1 = tf.keras.layers.Conv2D(16, (3, 3), activation='tanh', kernel_initializer='he_normal', padding='same')(c1)
p1 = tf.keras.layers.MaxPooling2D((2, 2))(c1)
f1 = tf.keras.layers.Lambda(lambda v: tf.signal.fft2d(tf.cast(v,tf.complex64)))(p1)
f1 = tf.keras.layers.Lambda(lambda v: tf.signal.ifft2d(tf.cast(v,tf.complex64)))(f1)
print(p1.shape)

c2 = tf.keras.layers.Conv2D(32, (3, 3), activation='tanh', kernel_initializer='he_normal', padding='same')(abs(f1))
c2 = channel_attention_module(c2)
c2 = spatial_attention_module(c2)
c2 = tf.keras.layers.Dropout(0.1)(c2)
c2 = tf.keras.layers.Conv2D(32, (3, 3), activation='tanh', kernel_initializer='he_normal', padding='

# continue 