# import the necessary packages
from keras import backend as K
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation, Dropout
from keras.layers.core import Dense

from keras.layers.core import Flatten
from keras.models import Sequential

class XRAY:
    @staticmethod
    def build(width, height, depth, classes):
        # initialize the model
        model = Sequential()
        inputShape = (height, width, depth) # (64, 64, 1) since the pictures are in greyscale

        # if we are using "channels first", update the input shape
        if K.image_data_format() == "channels_first":
            inputShape = (depth, height, width)

        # (8 filters, kernel_size = (2, 2), zero_padding, input_shape)
        model.add(Conv2D(8, (2, 2), padding="zero_padding",
                         input_shape=inputShape))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
 
        model.add(Conv2D(100, (2, 2), padding="same"))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        model.add(Dropout(0.2))

        model.add(Conv2D(200, (2, 2), padding="same"))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        model.add(Flatten())
        model.add(Dense(50))
        model.add(Activation("relu"))

        # softmax classifier
        model.add(Dense(classes)) # number of classes == 2
        model.add(Activation("softmax"))

        # return the constructed network architecture
        return model
