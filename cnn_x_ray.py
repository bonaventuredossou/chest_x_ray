# import the necessary packages
from keras import backend as K
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation, Dropout
from keras.layers.core import Dense

from keras.layers.core import Flatten
from keras.models import Sequential

class FSER20_1:
    @staticmethod
    def build(width, height, depth, classes):
        # initialize the model
        model = Sequential()
        inputShape = (height, width, depth) # (64, 64, 1)

        # if we are using "channels first", update the input shape
        if K.image_data_format() == "channels_first":
            inputShape = (depth, height, width)

        # first set of CONV => RELU => POOL layers
        model.add(Conv2D(8, (2, 2), padding="zero_padding",
                         input_shape=inputShape))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
 
        # second set of CONV => RELU => POOL layers
        model.add(Conv2D(100, (2, 2), padding="same"))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        model.add(Dropout(0.2))

        # third set of CONV => RELU => POOL layers
        model.add(Conv2D(200, (2, 2), padding="same"))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        # Trying the idea of Independent component
        # model.add(Dropout(0.2))

        # first (and only) set of FC => RELU layers
        model.add(Flatten())
        model.add(Dense(50))
        model.add(Activation("relu"))

        # softmax classifier
        model.add(Dense(classes)) # number of classes == 2
        model.add(Activation("softmax"))

        # return the constructed network architecture
        return model
