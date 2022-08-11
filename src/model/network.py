from keras.layers import Flatten, Dense, Conv2D, Input, Dropout, Conv1D
from keras.models import Model, Sequential
from keras.initializers import GlorotNormal
from tensorflow.keras import backend as K
import tensorflow as tf


# def create_model(window_size, feature_dim, kernel_size, filter_num, dropout_rate):
#     """
#     create a Deep convolutional neural network using models.sequential

#     window_size: Integer, at each time t sensor values [t-window_size+1, t] are included 
#     feature_dim: Integer, # of features (e.g. sensors) to be included
#     kernel_size: Tuple (filter length, filter width). E.g. (10, 1)
#                  only for the first 4 convolutional layers
#     filter_num: Integer, # of filters
#                  only for the first 4 convolutional layers
#     dropout_rate: Float, dropout_rate to be used in the dropout layer             
#     """
#     model = Sequential()
#     model.add(Conv2D(filters=filter_num, kernel_size=kernel_size, padding='same', activation='tanh', kernel_initializer=GlorotNormal(), input_shape=(window_size, feature_dim, 1)))
#     model.add(Conv2D(filters=filter_num, kernel_size=kernel_size, padding='same', activation='tanh', kernel_initializer=GlorotNormal()))
#     model.add(Conv2D(filters=filter_num, kernel_size=kernel_size, padding='same', activation='tanh', kernel_initializer=GlorotNormal()))
#     model.add(Conv2D(filters=filter_num, kernel_size=kernel_size, padding='same', activation='tanh', kernel_initializer=GlorotNormal()))
#     model.add(Conv2D(filters=1, kernel_size=(3, 1), padding='same', activation='tanh'))
#     model.add(Flatten())
#     model.add(Dropout(dropout_rate))
#     model.add(Dense(100, activation='tanh'))
#     model.add(Dense(1))

#     return model 



def create_model(window_size, feature_dim, kernel_size, filter_num, dropout_rate):
    """
    create a Deep convolutional neural network using models.Model(inputs, outputs)

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
    drop = Dropout(dropout_rate)(flat)
    hidden = Dense(100, activation='tanh')(drop)
    outputs = Dense(1)(hidden)
    model = Model(inputs=inputs, outputs=outputs)

    return model 


def create_MQDCNN(quantiles, window_size, feature_dim, kernel_size, filter_num, dropout_rate):
    """
    create Deep convolutional neural networks for multiple quantiles using models.Model(inputs, outputs)

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
    drop = Dropout(dropout_rate)(flat)
    hidden = Dense(100, activation='tanh')(drop)
    outputs = [Dense(1, name="q%d" % q_i)(hidden) for q_i in range(output_dim)]
    model = Model(inputs=inputs, outputs=outputs)

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




class MultiQuantileLoss(tf.keras.losses.Loss):
    """
    https://github.com/adlzanchetta/quantile-regression-tensorflow.git
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



# def create_MQDCNN(quantiles:list, training_x_values:np.ndarray, internal_nodes:list = [32, 32],
#                model_name:str = "mqnn", optimizer=None, input_normalization:bool = True):
#     """
#     Builds a multi quantile deep convolutional neural network
#     :param quantiles: List of floats with quantiles to be trained. E.g. [0.25, 0.50, 0.75]
#     :param training_x_values: 2-D numpy array with form [n-records, n-features]. NO categorical data expected.
#     :param internal_nodes: List of integers describing internal nodes. E.g. [24, 12] means: two dense layers with 24 and 12 nodes, respectively.
#     :param model_name: String to be used as model name. Default: 'mqnn'.
#     :param optimizer: A tf.optimizers.Optimizer object to be used as optimizer. If not given, uses default Adam with training step of 0.001.
#     :param input_normalization: Boolean. If True (default) includes a normalization step built in to the NN.
#     """
    
#     input_dim = training_x_values.shape[1]
#     output_dim = len(quantiles)
    
#     # define normalizer
#     normalizer = preprocessing.Normalization()
#     normalizer.adapt(training_x_values)
    
#     # build model's node structure
#     inputs = layers.Input(shape=input_dim)
#     mdl = normalizer(inputs)
#     for n_nodes in internal_nodes:
#         mdl = layers.Dense(n_nodes, activation='relu')(mdl)
#     outputs = [layers.Dense(1, activation='linear', name="q%d" % q_i)(mdl) for q_i in range(output_dim)]
#     del input_dim, output_dim, mdl, normalizer
    
#     # define optimizer and loss functions
#     optm_func = tf.optimizers.Adam(learning_rate=0.001) if optimizer is None else optimizer
#     loss_func = MultiQuantileLoss(quantiles=quantiles)
    
#     # build and compile model
#     model = tf.keras.models.Model(inputs=inputs, outputs=outputs, name=model_name)
#     model.compile(optimizer=optm_func, loss=loss_func)
    
#     return model