import time
import tensorflow as tf
import numpy as np
import pandas as pd



from keras.models import Model, load_model
from keras.layers import Input, Dense
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras import regularizers

from csv_utils import CSV


RANDOM_SEED = 1011

class Autoencoder():
    def __init__(self, input_dim):
        self.input_dim = input_dim 
        self.model = self.build_model(6)

    def build_model(self , encoding_dim):
        input_layer = Input(shape=(self.input_dim, ))
        encoder = Dense(encoding_dim, activation="tanh",activity_regularizer=regularizers.l1(10e-5))(input_layer)
        encoder = Dense(int(encoding_dim / 2), activation="tanh")(encoder)
        encoder = Dense(int(2), activation="tanh")(encoder)
        decoder = Dense(int(encoding_dim/ 2), activation='tanh')(encoder)
        decoder = Dense(int(encoding_dim), activation='tanh')(decoder)
        decoder = Dense(self.input_dim, activation='tanh')(decoder)
        autoencoder = Model(inputs=input_layer, outputs=decoder)
        autoencoder.compile(optimizer='adam', loss='mse' )
        autoencoder.summary()   
        return autoencoder 


    def prepare_data(self):
        df = CSV("sensor_data.csv")
        df.normalaize()
        return df.to_train_test()

        

    def __call__(self , epochs , batch_size):
        X_train_scaled, X_test_scaled = self.prepare_data()
        tensorboard = TensorBoard(log_dir=f'./logs/{time.time()}',
                                                    batch_size=batch_size,
                                                    write_graph=True,
                                                    histogram_freq=5,
                                                    write_images=True,
                                                    write_grads=True)
        checkpointer  = ModelCheckpoint(filepath='weights/autoencoder.best.hdf5', verbose = 1)
        self.history = self.model.fit(X_train_scaled, X_train_scaled,
                        epochs=epochs,
                        batch_size=batch_size,
                        shuffle=True,
                        validation_split=0.2,
                        #verbose=0,
                        callbacks=[tensorboard,checkpointer] )
        """ print(history.history)     
        predictions = self.model.predict(X_test_scaled)   
        mse = np.mean(np.power(X_test_scaled - predictions, 2), axis=1)
        return pd.DataFrame({'reconstruction_error': mse})      """  



""" model = Autoencoder(9)
score = model(epochs=100 , batch_size=128) """


outliers = score.index[score.reconstruction_error > 0.1].tolist()
print(outliers)