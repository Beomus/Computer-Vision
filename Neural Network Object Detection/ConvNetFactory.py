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
            "shallownet": ConvNetFactory.ShallowNet
            #"lenet": ConvNetFactory.Letnet,
            #"karpathynet": ConvNetFactory.KarpathyNet,
            #"minivggnet": ConvNetFactory.MiniVGGNet
        }

        builder = mappings.get(name, None)

        if builder is None:
            return None

        return builder(*args, **kwargs)

    @staticmethod
    def ShallowNet(numChannels, imgRows, imgCols, numClasses, **kwargs):
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

    
