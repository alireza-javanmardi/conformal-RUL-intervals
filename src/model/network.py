from keras.layers import Flatten, Dense, Conv2D, Input, Dropout, Conv1D
from keras import Model, models, layers
from keras.initializers import GlorotNormal
from tensorflow.keras import backend as K
import tensorflow as tf

def create_model(window_size, feature_dim, kernel_size, filter_num, dropout_rate):
    
    model = models.Sequential()
    model.add(layers.Conv2D(filters=filter_num, kernel_size=kernel_size, padding='same', activation='tanh', kernel_initializer=GlorotNormal(), input_shape=(window_size, feature_dim, 1)))
    model.add(layers.Conv2D(filters=filter_num, kernel_size=kernel_size, padding='same', activation='tanh', kernel_initializer=GlorotNormal()))
    model.add(layers.Conv2D(filters=filter_num, kernel_size=kernel_size, padding='same', activation='tanh', kernel_initializer=GlorotNormal()))
    model.add(layers.Conv2D(filters=filter_num, kernel_size=kernel_size, padding='same', activation='tanh', kernel_initializer=GlorotNormal()))
    model.add(layers.Conv2D(filters=1, kernel_size=(3, 1), padding='same', activation='tanh'))
    model.add(layers.Flatten())
    model.add(layers.Dropout(dropout_rate))
    model.add(layers.Dense(100))
    model.add(layers.Dense(1))

    return model



def quantile_loss(q,y_true,y_pred):
    """
    q -- quantile level
    y_true -- true values
    y_pred -- predicted values
    """
    diff = (y_true - y_pred)
    mask = y_true >= y_pred
    mask_ = y_true < y_pred
    loss = (q * K.sum(tf.boolean_mask(diff, mask), axis=-1) - (1 - q) * K.sum(tf.boolean_mask(diff, mask_), axis=-1))

    return loss