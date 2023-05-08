import tensorflow as tf
import numpy as np


def predict(path, padded_sequence):
    model = tf.keras.models.load_model(path)
    output = model.predict(padded_sequence[0:1])
    output = np.where(output<0.5, 1, 0)
    return output[0]