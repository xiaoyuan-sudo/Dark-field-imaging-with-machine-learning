# import the necessary packages
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras.layers.convolutional import ZeroPadding2D
from keras import backend as K

class MyAlexNet:
    def build(width, height, depth, classes):
		# initialize the model along with the input shape to be
		# "channels last" and the channels dimension itself
        model = Sequential()
        inputShape = (height, width, depth)
        chanDim = -1

		# if we are using "channels first", update the input shape
		# and channels dimension
        if K.image_data_format() == "channels_first":
            inputShape = (depth, height, width)
            chanDim = 1

					# CONV => RELU => POOL
        model.add(Conv2D(96, (11, 11), strides=(4,4),input_shape=inputShape))
        model.add(Activation("relu"))
        model.add(BatchNormalization())

        model.add(Conv2D(96, (3, 3), strides=(2,2),padding='same'))
        model.add(Activation("relu"))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(3, 3), strides=(2,2)))
        model.add(Dropout(0.25))

        model.add(Conv2D(256, (3, 3), strides=(1,1), padding='same'))
        model.add(Activation("relu"))
        model.add(BatchNormalization())
        model.add(Conv2D(256, (3, 3), strides=(1,1), padding='same'))
        model.add(Activation("relu"))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(3, 3),strides=(2,2)))
        model.add(Dropout(0.25))   

        model.add(Flatten())
        model.add(Dense(1024))
        model.add(Activation("relu"))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))

        model.add(Dense(classes))
        model.add(Activation("softmax"))

		# return the constructed network architecture
        return model
