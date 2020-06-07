import keras
import tensorflow as tf

class ConvNetFactory:
    def __init__(self):
        pass

    @staticmethod
    def build(name, *args, **kwargs):
        """
        parameters:
            name: name of the network architecture that we want to use
            *args: positional arguments
            **kwargs: keyword arguments
        """
        # define the network mappings
        mappings = {
            "shallownet": ConvNetFactory.ShallowNet,
            "lenet": ConvNetFactory.LeNet,
            "karpathynet": ConvNetFactory.KarpathyNet,
            "minivggnet": ConvNetFactory.MiniVGGNet
        }

        builder = mappings.get(name, None)

        if builder is None:
            return None

        return builder(*args, **kwargs)

    @staticmethod
    def ShallowNet(numChannels, imgRows, imgCols, numClasses, **kwargs):
        """
        Model architecture: INPUT => CONV2D => RELU => OUTPUT (DENSE => SOFTMAX)
        """
        model = keras.models.Sequential()
        input_shape = (imgRows, imgCols, numChannels)

        # switch between Theano vs Tensorflow backends where Theano uses channel first and Tensorflow use channel last
        if tf.keras.backend.image_data_format() == "channel_first" :
            input_shape = (numChannels, imgRows, imgCols)

        model.add(keras.layers.Conv2D(32, (3, 3), padding="same", input_shape=input_shape))
        model.add(keras.layers.Activation('relu'))

        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(numClasses))
        model.add(keras.layers.Activation("softmax"))

        return model

    @staticmethod
    def LeNet(numChannels, imgRows, imgCols, numClasses, activation="relu", **kwargs):
        """
        Model architecture: INPUT => (CONV2D => RELU => MAXPOOLING2D) * 2 => DENSE => RELU => OUTPUT (DENSE => SOFTMAX)
        """
        model = keras.models.Sequential()
        input_shape = (imgRows, imgCols, numChannels)

        if tf.keras.backend.image_data_format() == "channel_first" :
            input_shape = (numChannels, imgRows, imgCols)

        model.add(keras.layers.Conv2D(32, (3, 3), padding="same", input_shape=input_shape))
        model.add(keras.layers.Activation(activation))
        model.add(keras.layers.MaxPooling2D((2, 2)))

        model.add(keras.layers.Conv2D(64, (3, 3), padding="same"))
        model.add(keras.layers.Activation(activation))
        model.add(keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(256))
        model.add(keras.layers.Activation(activation))
        model.add(keras.layers.Dense(numClasses))
        model.add(keras.layers.Activation("softmax"))

        return model

    @staticmethod
    def KarpathyNet(numChannels, imgRows, imgCols, numClasses, dropout=False, **kwargs):
        """
        Model architecture: INPUT => (CONV2D => RELU => MAXPOOLING2D => [DROPOUT]) * 3 => OUTPUT (DENSE => SOFTMAX)
        """
        input_shape = (numChannels, imgRows, imgCols) if tf.keras.backend.image_data_format() == "channel_first" else (imgRows, imgCols, numChannels)
        model = keras.models.Sequential()
        model.add(keras.layers.Conv2D(16, (5, 5), padding="same", input_shape=input_shape, activation='relu'))
        model.add(keras.layers.MaxPooling2D((2, 2), (2, 2)))
        if dropout:
            model.add(keras.layers.Dropout(0.2))

        model.add(keras.layers.Conv2D(32, (5, 5), padding="same", input_shape=input_shape, activation='relu'))
        model.add(keras.layers.MaxPooling2D((2, 2), (2, 2)))
        if dropout:
            model.add(keras.layers.Dropout(0.2))

        model.add(keras.layers.Conv2D(32, (5, 5), padding="same", input_shape=input_shape, activation='relu'))
        model.add(keras.layers.MaxPooling2D((2, 2), (2, 2)))
        if dropout:
            model.add(keras.layers.Dropout(0.2))

        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(numClasses))
        model.add(keras.layers.Activation('softmax'))

        return model

    @staticmethod 
    def MiniVGGNet(numChannels, imgRows, imgCols, numClasses, dropout=False, **kwargs):
        """
        Model architecture: INPUT => (CONV2D * 2 => POOLING) * 2 => DENSE * 2 => OUTPUT
        """
        input_shape = (numChannels, imgRows, imgCols) if tf.keras.backend.image_data_format() == "channel_first" else (imgRows, imgCols, numChannels)

        model = keras.models.Sequential()
        model.add(keras.layers.Conv2D(32, (3, 3), padding='same', input_shape=input_shape, activation='relu'))
        model.add(keras.layers.Conv2D(32, (3, 3), activation='relu'))
        model.add(keras.layers.MaxPooling2D((2, 2)))
        if dropout:
            model.add(keras.layers.Dropout(0.2))

        model.add(keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu'))
        model.add(keras.layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(keras.layers.MaxPooling2D((2, 2)))
        if dropout:
            model.add(keras.layers.Dropout(0.2))

        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(512, activation='relu'))
        if dropout:
            model.add(keras.layers.Dropout(0.2))

        model.add(keras.layers.Dense(numClasses, activation='softmax'))

        return model
