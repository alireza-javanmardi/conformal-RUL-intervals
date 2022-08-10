from keras.layers import Flatten, Dense, Conv2D, Input, Dropout, Conv1D
from keras import Model, models, layers
from keras.initializers import GlorotNormal
def create_model(window_size, feature_dim, kernel_size, filter_num, dropout_rate):

    input_layer = Input(shape=(window_size, feature_dim, 1), )
    #conv layer 1
    out = Conv2D(filters=filter_num, kernel_size=kernel_size, padding='same', activation='tanh')(input_layer)
    #conv layer 2
    out = Conv2D(filters=filter_num, kernel_size=kernel_size, padding='same', activation='tanh')(out)
    #conv layer 3
    out = Conv2D(filters=filter_num, kernel_size=kernel_size, padding='same', activation='tanh')(out)
    #conv layer 4
    out = Conv2D(filters=filter_num, kernel_size=kernel_size, padding='same', activation='tanh')(out)
    #conv layer 5
    out = Conv2D(filters=1, kernel_size=(3, 1), padding='same', activation='tanh')(out)
    #flatten layer 
    out = Flatten()(out)
    #dropout layer 
    out = Dropout(dropout_rate)(out)
    out = Dense(1, activation="tanh")(out)
    model = Model(inputs=input_layer, outputs=out)

    return model



def simple_cnn(window_size, feature_dim, kernel_size, filter_num, dropout_rate):
    
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