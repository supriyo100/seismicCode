"""UNet++ implementation with attention blocks"""

import tensorflow as tf

def conv_block(inputs, num_filters, dropout_prob=0.1):
    x = tf.keras.layers.Conv2D(num_filters, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(dropout_prob)(x)
    x = tf.keras.layers.Conv2D(num_filters, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    return x

def upsample_block(inputs, num_filters):
    x = tf.keras.layers.UpSampling2D((2, 2))(inputs)
    x = tf.keras.layers.Conv2D(num_filters, (2, 2), activation='relu', padding='same', kernel_initializer='he_normal')(x)
    return x

def attention_block(inputs, skip):
    x = tf.keras.layers.Conv2D(filters=skip.shape[-1] // 2, kernel_size=1, activation='relu', padding='same')(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2D(filters=skip.shape[-1] // 2, kernel_size=1, activation='relu', padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2D(filters=skip.shape[-1], kernel_size=1, activation='sigmoid', padding='same')(x)
    x = tf.keras.layers.multiply([x, skip])
    return x

def nested_unet(input_shape, num_filters=16, dropout_prob=0.1):
    inputs = tf.keras.layers.Input(input_shape)

    # Encoder path
    conv1 = conv_block(inputs, num_filters=num_filters)
    pool1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = conv_block(pool1, num_filters=num_filters*2)
    pool2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = conv_block(pool2, num_filters=num_filters*4)
    pool3 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = conv_block(pool3, num_filters=num_filters*8)
    pool4 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv4)

    # Bottom
    conv5 = conv_block(pool4, num_filters=num_filters*16)
    up5 = upsample_block(conv5, num_filters=num_filters*8)

    # Decoder path
    att4 = attention_block(up5, conv4)
    concat4 = tf.keras.layers.concatenate([up5, att4], axis=-1)
    conv6 = conv_block(concat4, num_filters=num_filters*8)

    up6 = upsample_block(conv6, num_filters=num_filters*4)
    att3 = attention_block(up6, conv3)
    concat3 = tf.keras.layers.concatenate([up6, att3], axis=-1)
    conv7 = conv_block(concat3, num_filters=num_filters*4)

    up7 = upsample_block(conv7, num_filters=num_filters*2)
    att2 = attention_block(up7, conv2)
    concat2 = tf.keras.layers.concatenate([up7, att2], axis=-1)
    conv8 = conv_block(concat2, num_filters=num_filters*2)

    up8 = upsample_block(conv8
