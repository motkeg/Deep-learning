import numpy as np
import pandas as pd
import tensorflow as tf

from data import data_utils
from tensorflow.python import keras
from tensorflow.python.keras.models import Sequential , load_model # model type
from tensorflow.python.keras.layers import Dense ,Conv2D , Flatten # layers type

train,test = data_utils.Create_scale_data_files('data/train.csv', 'data/test.csv',True) # create scale dtda files
#train , test = data_utils.get_df() # get dataFrame from 'data_utils'

X = train.drop('total_earnings' , axis=1)
y = train[['total_earnings']]

X_test = test.drop('total_earnings' , axis=1)
y_test = test[['total_earnings']]

#define model
model = Sequential()
model.add(Dense(50,input_dim=9, activation='relu',name='L-1'))
model.add(Dense(100,activation='relu',name='L-2'))
model.add(Dense(50,activation='relu',name='L-3'))
model.add(Dense(1,name='Output'))

# compile model
model.compile(loss='mse' , optimizer='adam',metrics=['accuracy'])

# make a logger to use with tensorboard
tensorboard = keras.callbacks.TensorBoard(log_dir='logs/vanila',
                                          write_graph=True,
                                          histogram_freq=5,
                                          write_images=True)

model.fit(X , y,
          epochs=50,
          validation_split=0.25,
          shuffle=True,
          verbose=2,
          callbacks=[tensorboard])

#model = load_model('models/vanila.h5')

#export model to use on GCP
data_utils.export_model_to_GCP('vanila_ext' ,
                                model.input,
                                model.output,
                                keras.backend.get_session())

error_rate = model.evaluate(X_test,y_test)
print("MSE: " ,error_rate)

# save model
model.save('models/vanila.h5')
print("model saved in 'models' folder...")


# predict = model.predict(?)






