#import datapre

from csv_utils import CSV
from keras.models import  load_model
import numpy as np
import pandas as pd

""" df  =  CSV("creditcard.csv")
scaled = df.normalized()
print(scaled[:5])   """


df  = CSV("sensor_data.csv")
print(df.df.shape)

#df.normalaize()



model  = load_model("weights/autoencoder.best.hdf5")
x,y = df.to_train_test()
predictions = model.predict(x)

mse = np.mean(np.power(x - predictions, 2), axis=1)
df_error = pd.DataFrame({'reconstruction_error': mse}) 
print(df_error.describe())
outliers = df_error.index[df_error.reconstruction_error > 0.5].tolist()
print(outliers[:10])

