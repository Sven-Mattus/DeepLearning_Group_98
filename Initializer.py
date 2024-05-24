from enum import Enum
import tensorflow as tf

class Initializer(Enum):
    GN = tf.keras.initializers.GlorotNormal()
    GU = tf.keras.initializers.GlorotUniform()
