"""
this is an model that only use to detact fashion_mnist images
using tensorflow and kears
"""

import tensorflow as tf
import matplotlib.pyplot as plt 
import numpy as np

from tensorflow.keras.layers import (MaxPool2D , Conv2D , Activation, 
                                     Dropout , Flatten , 
                                     Dense , BatchNormalization)
from tensorflow.keras.models import Sequential
from tensorflow.keras.datasets import fashion_mnist 
from tensorflow.keras.callbacks import TensorBoard , ModelCheckpoint


flags = tf.app.flags
FLAGS = flags.FLAGS

USE_TPU = False
LABEL_NAMES = ['t_shirt', 'trouser', 'pullover', 'dress', 'coat', 'sandal', 'shirt', 'sneaker', 'bag', 'ankle_boots']

SERVICE_ACCOUNT_FILE = 'c:/Users/USER/msc-project-owner.json'


class cnn_fashion():

    # define tyhe train test and eval data
   

    def __init__(self):
        (self.x_train, self.y_train) ,(self.x_test, self.y_test)  = fashion_mnist.load_data()
        (self.x_valid,self.y_valid) = None, None
        self.model = Sequential()
        self.model_build()
        self.model.summary()
        self.tensorboard = TensorBoard(log_dir=f'./{FLAGS.log_dir}',
                          batch_size=FLAGS.batch_size,
                            write_graph=True,
                            histogram_freq=3,
                            write_images=True,
                            write_grads=True)

        self.checkpointer = ModelCheckpoint(filepath=f'./{FLAGS.job_dir}/fashion_mnist.weights.best.hdf5', verbose = 1, save_best_only=True)   
            
        print("x_train shape:", self.x_train.shape, "y_train shape:", self.y_train.shape)

    def __call__(self,train=True,predict=None):
        if train:
            self.prepare_data()
            self.Train()
        else :
            if predict is not None:
                self.Predict(predict)




    def prepare_data(self):
        # data Normalization
        self.x_train = self.x_train.astype('float32') / 255
        self.x_test = self.x_test.astype('float32') / 255
        # create a validation set fron the train data
        val_size  = int(len(self.x_train)*0.1)
        (self.x_valid,self.y_valid) = self.x_train[:val_size] , self.y_train[:val_size]
        #(self.x_tarin,self.y_train) = self.x_train[val_size:] , self.y_train[val_size:]

        # reshape the data to fit the model
        self.x_train = self.x_train.reshape(self.x_train.shape[0], 28, 28, 1)
        self.x_valid = self.x_valid.reshape(self.x_valid.shape[0], 28, 28, 1)
        self.x_test = self.x_test.reshape(self.x_test.shape[0], 28, 28, 1)

        # One-hot encode the labels
        self.y_train = tf.keras.utils.to_categorical(self.y_train, 10)
        self.y_valid = tf.keras.utils.to_categorical(self.y_valid, 10)
        self.y_test = tf.keras.utils.to_categorical(self.y_test, 10)


    def model_build(self):

        #-------------------------------
        # define the model and layers
        #------------------------------
        
        '''self.model.add(Conv2D(filters=64 , kernel_size =3 , 
                        padding="same" , activation="relu",
                        input_shape = (28,28,1)))
        self.model.add(MaxPool2D(pool_size=3))
        self.model.add(Dropout(0.3))                 
        self.model.add(Conv2D(filters=32 , kernel_size =3 , strides=2,
                        padding="same" , activation="relu"))
        self.model.add(MaxPool2D(pool_size=3))                 
        self.model.add(Flatten())
        self.model.add(Dense(256,activation="relu"))
        self.model.add(Dropout(0.4))
        self.model.add(Dense(10,activation="softmax"))'''
        with tf.variable_scope("cnn"):
            self.model.add(BatchNormalization(input_shape=(28,28,1)))
            self.model.add(Conv2D(64, (5, 5), padding='same', activation='relu'))
            self.model.add(MaxPool2D(pool_size=(2, 2), strides=(2,2)))
            self.model.add(Dropout(0.25))

            self.model.add(BatchNormalization(input_shape=(28,28,1)))
            self.model.add(Conv2D(128, (5, 5), padding='same', activation='relu'))
            self.model.add(MaxPool2D(pool_size=(2, 2)))
            self.model.add(Dropout(0.25))

            self.model.add(BatchNormalization(input_shape=(28,28,1)))
            self.model.add(Conv2D(256, (5, 5), padding='same', activation='relu'))
            self.model.add(MaxPool2D(pool_size=(2, 2), strides=(2,2)))
            self.model.add(Dropout(0.25))

            self.model.add(Flatten())
            self.model.add(Dense(256))
            self.model.add(Activation('relu'))
            self.model.add(Dropout(0.5))
            self.model.add(Dense(10))
            self.model.add(Activation('softmax'))

        if USE_TPU:
            #credentials = GoogleCredentials.from_stream(SERVICE_ACCOUNT_FILE) 
            #credentials, project = google.auth.from_stream(SERVICE_ACCOUNT_FILE)
            strategy = tf.contrib.tpu.TPUDistributionStrategy(
                        tf.contrib.cluster_resolver.TPUClusterResolver(tpu='grpc://10.240.1.2:8470' , zone="us-central1-b"))
                                                                       #credentials=credentials))
            self.model = tf.contrib.tpu.keras_to_tpu_model(self.model, strategy=strategy)

        #-------------------------------
        # compile the model 
        #-----------------------------
        self.model.compile(loss='categorical_crossentropy',
                    optimizer='adam',
                    metrics=['accuracy'])


    def Train(self):
        #-------------------------------
        # fit (train) the model 
        #-----------------------------
        self.model.fit(self.x_train, self.y_train,
                    batch_size=FLAGS.batch_size,
                    epochs=FLAGS.epochs,
                    validation_data=(self.x_valid,self.y_valid),
                    #validation_steps=1,
                    callbacks=[self.tensorboard,self.checkpointer],
                    shuffle=True)

        # Evaluate the model on test set
        score = self.model.evaluate(self.x_test, self.y_test, verbose=0)
        # Print test accuracy
        print('\n', 'Test accuracy:', score[1])  
        '''if score[1] >=0.9:
            self.model.save("./fashion_cnn/weights/fashion_%2f.hdf5" % score[1]) 
            print("model saved in 'weights' folder...")     ''' 

    def Predict(self,predict):
        pass
        #TODO: complate this phase  
        # 
        # 
        # 
    
def plot_predictions(images, predictions):
        n = images.shape[0]
        nc = int(np.ceil(n / 4))
        f, axes = plt.subplots(nc, 4)
        for i in range(nc * 4):
            y = i // 4
            x = i % 4
            axes[x, y].axis('off')
            
            label = LABEL_NAMES[np.argmax(predictions[i])]
            confidence = np.max(predictions[i])
            if i > n:
                continue
            axes[x, y].imshow(images[i])
            axes[x, y].text(0.5, 0.5, label + '\n%.3f' % confidence, fontsize=14)

        plt.gcf().set_size_inches(8, 8)      
               