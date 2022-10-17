import tensorflow as tf
from tensorflow.keras.layers import Flatten, Dense, Conv2D, Input, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.initializers import GlorotNormal



def create_DCNN(window_size, feature_dim, kernel_size, filter_num, dropout_rate):
    """create a Deep convolutional neural network using models.Model(inputs, outputs)

    window_size: Integer, at each time t sensor values [t-window_size+1, t] are included 
    feature_dim: Integer, # of features (e.g. sensors) to be included
    kernel_size: Tuple (filter length, filter width). E.g. (10, 1)
                 only for the first 4 convolutional layers
    filter_num: Integer, # of filters
                 only for the first 4 convolutional layers
    dropout_rate: Float, dropout_rate to be used in the dropout layer             
    """
    inputs = Input(shape=(window_size, feature_dim, 1))
    conv1 = Conv2D(filters=filter_num, kernel_size=kernel_size, padding='same', activation='tanh', kernel_initializer=GlorotNormal())(inputs)
    conv2 = Conv2D(filters=filter_num, kernel_size=kernel_size, padding='same', activation='tanh', kernel_initializer=GlorotNormal())(conv1)
    conv3 = Conv2D(filters=filter_num, kernel_size=kernel_size, padding='same', activation='tanh', kernel_initializer=GlorotNormal())(conv2)
    conv4 = Conv2D(filters=filter_num, kernel_size=kernel_size, padding='same', activation='tanh', kernel_initializer=GlorotNormal())(conv3)
    conv5 = Conv2D(filters=1, kernel_size=(3, 1), padding='same', activation='tanh', kernel_initializer=GlorotNormal())(conv4)
    flat = Flatten()(conv5)
    drop = Dropout(rate=dropout_rate)(flat)
    hidden = Dense(100, activation='tanh', kernel_initializer=GlorotNormal())(drop)
    outputs = Dense(1, kernel_initializer=GlorotNormal())(hidden)
    model = Model(inputs=inputs, outputs=outputs)

    return model 


def create_MQDCNN(quantiles, window_size, feature_dim, kernel_size, filter_num, dropout_rate):
    """create Deep convolutional neural networks for multiple quantiles using models.Model(inputs, outputs)

    quantiles: List of floats with quantiles to be trained. E.g. [0.25, 0.50, 0.75]
    window_size: Integer, at each time t sensor values [t-window_size+1, t] are included 
    feature_dim: Integer, # of features (e.g. sensors) to be included
    kernel_size: Tuple (filter length, filter width). E.g. (10, 1)
                 only for the first 4 convolutional layers
    filter_num: Integer, # of filters
                 only for the first 4 convolutional layers
    dropout_rate: Float, dropout_rate to be used in the dropout layer             
    """
    output_dim = len(quantiles)
    inputs = Input(shape=(window_size, feature_dim, 1))
    conv1 = Conv2D(filters=filter_num, kernel_size=kernel_size, padding='same', activation='tanh', kernel_initializer=GlorotNormal())(inputs)
    conv2 = Conv2D(filters=filter_num, kernel_size=kernel_size, padding='same', activation='tanh', kernel_initializer=GlorotNormal())(conv1)
    conv3 = Conv2D(filters=filter_num, kernel_size=kernel_size, padding='same', activation='tanh', kernel_initializer=GlorotNormal())(conv2)
    conv4 = Conv2D(filters=filter_num, kernel_size=kernel_size, padding='same', activation='tanh', kernel_initializer=GlorotNormal())(conv3)
    conv5 = Conv2D(filters=1, kernel_size=(3, 1), padding='same', activation='tanh', kernel_initializer=GlorotNormal())(conv4)
    flat = Flatten()(conv5)
    drop = Dropout(rate=dropout_rate)(flat)
    hidden = Dense(100, activation='tanh', kernel_initializer=GlorotNormal())(drop)
    outputs = [Dense(1, kernel_initializer=GlorotNormal(), name="q%d" % q_i)(hidden) for q_i in range(output_dim)]
    model = Model(inputs=inputs, outputs=outputs)

    return model 


class MultiQuantileLoss(tf.keras.losses.Loss):
    """https://github.com/adlzanchetta/quantile-regression-tensorflow.git
    """
    
    def __init__(self, quantiles:list, **kwargs):
        super(MultiQuantileLoss, self).__init__(**kwargs)
        
        self.quantiles = quantiles

    def call(self, y_true, y_pred):
        
        # get quantile value
        q_id = int(y_pred.name.split("/")[1][1:])
        q = self.quantiles[q_id]
        
        # minimize quantile error
        q_error = tf.subtract(y_true, y_pred)
        q_loss = tf.reduce_mean(tf.maximum(q*q_error, (q-1)*q_error), axis=-1)
        return q_loss
