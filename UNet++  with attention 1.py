""" Unet++ with attention module"""

import tensorflow as tf

def attention_gate(input_1, input_2, n_filters):
    g1 = tf.keras.layers.Conv2D(n_filters, (1, 1))(input_1)
    x1 = tf.keras.layers.Conv2D(n_filters, (1, 1))(input_2)
    psi = tf.keras.layers.ReLU()(tf.keras.layers.Add()([g1, x1]))
    psi = tf.keras.layers.Conv2D(1, (1, 1))(psi)
    psi = tf.keras.layers.BatchNormalization()(psi)
    psi = tf.keras.layers.Activation('sigmoid')(psi)
    input_1 = tf.keras.layers.Multiply()([input_1, psi])
    return input_1

inputs = tf.keras.layers.Input((128, 128, 1))

# Contraction path
c1 = tf.keras.layers.Conv2D(16, (3, 3), activation='tanh', kernel_initializer='he_normal', padding='same')(inputs)
c1 = tf.keras.layers.Dropout(0.1)(c1)
c1 = tf.keras.layers.Conv2D(16, (3, 3), activation='tanh', kernel_initializer='he_normal', padding='same')(c1)
p1 = tf.keras.layers.MaxPooling2D((2, 2))(c1)
f1 = tf.keras.layers.Lambda(lambda v: tf.signal.fft2d(tf.cast(v,tf.complex64)))(p1)
f1 = tf.keras.layers.Lambda(lambda v: tf.signal.ifft2d(tf.cast(v,tf.complex64)))(f1)
print(p1.shape)

c2 = tf.keras.layers.Conv2D(32, (3, 3), activation='tanh', kernel_initializer='he_normal', padding='same')(abs(f1))
c2 = tf.keras.layers.Dropout(0.1)(c2)
c2 = tf.keras.layers.Conv2D(32, (3, 3), activation='tanh', kernel_initializer='he_normal', padding='same')(c2)
p2 = tf.keras.layers.MaxPooling2D((2, 2))(c2)
f2 = tf.keras.layers.Lambda(lambda v: tf.signal.fft2d(tf.cast(v,tf.complex64)))(p2)
f2 = tf.keras.layers.Lambda(lambda v: tf.signal.ifft2d(tf.cast(v,tf.complex64)))(f2)
print(p2.shape)


c3 = tf.keras.layers.Conv2D(64, (3, 3), activation='tanh', kernel_initializer='he_normal', padding='same')(abs(f2))
c3 = tf.keras.layers.Dropout(0.2)(c3)
c3 = tf.keras.layers.Conv2D(64, (3, 3), activation='tanh', kernel_initializer='he_normal', padding='same')(c3)
p3 = tf.keras.layers.MaxPooling2D((2, 2))(c3)
f3 = tf.keras.layers.Lambda(lambda v: tf.signal.fft2d(tf.cast(v,tf.complex64)))(p3)
f3 = tf.keras.layers.Lambda(lambda v: tf.signal.ifft2d(tf.cast(v,tf.complex64)))(f3)
print(p3.shape)


c4 = tf.keras.layers.Conv2D(128, (3, 3), activation='tanh', kernel_initializer='he_normal', padding='same')(abs(f3))
c4 = tf.keras.layers.Dropout(0.2)(c4)
c4 = tf.keras.layers.Conv2D(128, (3, 3), activation='tanh', kernel_initializer='he_normal', padding='same')(c4)
p4 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(c4)
f4 = tf.keras.layers.Lambda(lambda v: tf.signal.fft2d(tf.cast(v,tf.complex64)))(p4)
f4 = tf.keras.layers.Lambda(lambda v: tf.signal.ifft2d(tf.cast(v,tf.complex64)))(f4)
print(p4.shape)
 

c5 = tf.keras.layers.Conv2D(256, (3, 3), activation='tanh', kernel_initializer='he_normal', padding='same')(abs(f4))
c5 = tf.keras.layers.Dropout(0.3)(c5)
c5 = tf.keras.layers.Conv2D(256, (3, 3), activation='tanh', kernel_initializer='he_normal', padding='same')(c5)

# Expansive path
u6 = tf.keras.layers.Conv2DTranspose(128, (2, 2), strides=(2, 2),padding='same')(c5)
u6 = attention_gate(u6, c4, 128)
u6 = tf.keras.layers.Concatenate()([u6, c4])
c6 = tf.keras.layers.Conv2D(128, (3, 3), activation='tanh', kernel_initializer='he_normal', padding='same')(u6)
c6 = tf.keras.layers.Dropout(0.2)(c6)
c6 = tf.keras.layers.Conv2D(128, (3, 3), activation='tanh', kernel_initializer='he_normal', padding='same')(c6)

u7 = tf.keras.layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)
u7 = attention_gate(u7, c3, 64)
u7 = tf.keras.layers.Concatenate()([u7, c3])
c7 = tf.keras.layers.Conv2D(64, (3, 3), activation='tanh', kernel_initializer='he_normal', padding='same')(u7)
c7 = tf.keras.layers.Dropout(0.2)(c7)
c7 = tf.keras.layers.Conv2D(64, (3, 3), activation='tanh', kernel_initializer='he_normal', padding='same')(c7)

u8 = tf.keras.layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7)
u8 = attention_gate(u8, c2, 32)
u8 = tf.keras.layers.Concatenate()([u8, c2])
c8 = tf.keras.layers.Conv2D(32, (3, 3), activation='tanh', kernel_initializer='he_normal', padding='same')(u8)
c8 = tf.keras.layers.Dropout(0.1)(c8)
c8 = tf.keras.layers.Conv2D(32, (3, 3), activation='tanh', kernel_initializer='he_normal', padding='same')(c8)

u9 = tf.keras.layers.Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c8)
u9 = attention_gate(u9, c1, 16)
u9 = tf.keras.layers.Concatenate()([u9, c1])
c9 = tf.keras.layers.Conv2D(16, (3, 3), activation='tanh', kernel_initializer='he_normal', padding='same')(u9)
c9 = tf.keras.layers.Dropout(0.1)(c9)
c9 = tf.keras.layers.Conv2D(16, (3, 3), activation='tanh', kernel_initializer='he_normal', padding='same')(c9)

outputs = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid')(c9)

model = tf.keras.models.Model(inputs=[inputs], outputs=[outputs])
