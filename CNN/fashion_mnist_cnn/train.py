

import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist 
#from tensorflow.keras import initializers

import matplotlib.pyplot as plt
import numpy as np

from cnn_fashion_model import cnn_fashion , plot_predictions
from tensorflow.keras.models import load_model
tf.logging.set_verbosity(tf.logging.ERROR)


flags = tf.app.flags

flags.DEFINE_integer("epochs", 2, "Epoch to train [250]")
flags.DEFINE_integer("batch_size",128 , "The number of batch images [64]")
flags.DEFINE_string("log_dir" , 'logs' , "Directory name to save the logs for tensorboard [logs]")
flags.DEFINE_string("job_dir" , '../weights' , "Directory name to save the checkpoints and weights [checkpoints]")
FLAGS = flags.FLAGS




def main(_):


    model  = cnn_fashion()
    model()
    #(x_train, y_train) ,(_, _)  = fashion_mnist.load_data()
    '''model = load_model("./fashion_cnn/logs/fashion-cnn-weights.best.hdf5")
    

    plot_predictions(np.squeeze(x_train[:16]), 
                 model.predict(x_train[:16]))'''


if __name__ == '__main__':
    tf.app.run()    


        



