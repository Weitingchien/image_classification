import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dense, GlobalAveragePooling2D

def mlpconv_block(input, num_filters, filter_size, strides, name):
    conv = Conv2D(num_filters, filter_size, strides=strides, padding='same', activation='relu', name=name+'_conv')(input)
    conv = Conv2D(num_filters, (1,1), padding='same', activation='relu', name=name+'_conv1x1_1')(conv)
    conv = Conv2D(num_filters, (1,1), padding='same', activation='relu', name=name+'_conv1x1_2')(conv)
    return conv

def create_nin_model():
    input_img = Input(shape=(28, 28, 1))
    mlpconv1 = mlpconv_block(input_img, 32, (5,5), (1,1), 'mlpconv1')
    pool1 = MaxPooling2D((3,3), strides=(2,2), padding='same', name='pool1')(mlpconv1)
    mlpconv2 = mlpconv_block(pool1, 64, (5,5), (1,1), 'mlpconv2')
    pool2 = MaxPooling2D((3,3), strides=(2,2), padding='same', name='pool2')(mlpconv2)
    mlpconv3 = mlpconv_block(pool2, 128, (5,5), (1,1), 'mlpconv3')
    pool3 = GlobalAveragePooling2D(name='pool3')(mlpconv3)
    predictions = Dense(10, activation='softmax')(pool3)
    model = Model(inputs=input_img, outputs=predictions)
    return model

def main():
    model = create_nin_model()
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255
    x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255
    epochs = 10
    model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=epochs)

if __name__ == '__main__':
    main()
