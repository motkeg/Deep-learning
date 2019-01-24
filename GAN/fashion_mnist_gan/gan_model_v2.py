import os
import numpy as np
from tqdm import tqdm
import tensorflow as tf
# uncomment this to run on GPU
#tf.keras.backend.set_image_data_format('channels_first')
from tensorflow import keras

from utils import *



flags = tf.app.flags
FLAGS = flags.FLAGS


USE_TPU = False

from tensorflow.keras.layers import (Input, Dense, Reshape, Flatten, Dropout,
                                    BatchNormalization, Activation, ZeroPadding2D,
                                    LeakyReLU, UpSampling2D, Conv2D)

from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.optimizers import Adam

from tensorflow.keras import initializers


# Deterministic output.
# Tired of seeing the same results every time? Remove the line below.
np.random.seed(1000)


class DCGAN_V2():
    def __init__(self):
        
        # The results are a little better when the dimensionality of the random vector is only 10.
        # The dimensionality has been left at 100 for consistency with other GAN implementations.
        self.randomDim = 100

        # Load data
        (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
        X_train = (X_train.astype(np.float32) - 127.5)/127.5
        self.X_train = np.expand_dims(X_train, -1)
        #self.X_train = X_train[:, np.newaxis, :, :] # uncomment this to run on GPU


        self.tensorboard = keras.callbacks.TensorBoard(log_dir='./logs/V2',
                                                        batch_size=FLAGS.batch_size,
                                                        write_graph=True,
                                                        histogram_freq=0,
                                                        write_images=True,
                                                        write_grads=True)
        self.checkpointer  = keras.callbacks.ModelCheckpoint(filepath=f'{FLAGS.job_dir}/gan_model_v2.best.h5', verbose = 1)

        # Optimizer
        self.optimizer = Adam(lr=0.0002, beta_1=0.5)

        self.discriminator = self.build_D()
        self.generator = self.build_G()

        # Combined network
        self.discriminator.trainable = False
        ganInput = Input(shape=(self.randomDim,))
        x = self.generator(ganInput)
        ganOutput = self.discriminator(x)
        self.gan = Model(inputs=ganInput, outputs=ganOutput)
        self.gan.compile(loss='binary_crossentropy', optimizer=self.optimizer)

        self.dLosses = []
        self.gLosses = []


    def build_G(self):
        # Generator
        with tf.variable_scope("Generator"):
            generator = Sequential()
            generator.add(Dense(128*7*7, input_dim=self.randomDim, kernel_initializer=initializers.RandomNormal(stddev=0.02)))
            generator.add(LeakyReLU(0.2))
            generator.add(Reshape((7, 7,128))) # cange this to (128,7,7) if you run on GPU
            generator.add(UpSampling2D(size=(2, 2)))
            generator.add(Conv2D(64, kernel_size=(5, 5), padding='same'))
            generator.add(LeakyReLU(0.2))
            generator.add(UpSampling2D(size=(2, 2)))
            generator.add(Conv2D(1, kernel_size=(5, 5), padding='same', activation='tanh'))
            generator.compile(loss='binary_crossentropy', optimizer=self.optimizer)
            generator.summary()
        return generator


    def build_D(self):
        # Discriminator
        with tf.variable_scope("Discriminator"):
            discriminator = Sequential()
            discriminator.add(Conv2D(64, kernel_size=(5, 5), strides=(2, 2), padding='same', input_shape=(28, 28,1), kernel_initializer=initializers.RandomNormal(stddev=0.02)))
            discriminator.add(LeakyReLU(0.2))                                                # change this to (1,28,28) if you run on GPU
            discriminator.add(Dropout(0.3))
            discriminator.add(Conv2D(128, kernel_size=(5, 5), strides=(2, 2), padding='same'))
            discriminator.add(LeakyReLU(0.2))
            discriminator.add(Dropout(0.3))
            discriminator.add(Flatten())
            discriminator.add(Dense(1, activation='sigmoid'))
            discriminator.compile(loss='binary_crossentropy', optimizer=self.optimizer)
            discriminator.summary()
        return discriminator

        


    def __call__(self):
        epochs ,batchSize = FLAGS.epochs, FLAGS.batch_size
        batchCount = self.X_train.shape[0] // batchSize
        print (f'Epochs:{epochs}\nBatch size: {batchSize}\t | Batches per epoch: {batchCount}')
            
        for e in range(1, epochs+1):
            print ('-'*15, 'Epoch %d' % e, '-'*15)
            for _ in tqdm(range(batchCount)):
                # Get a random set of input noise and images
                noise = np.random.normal( 0,1, size=[batchSize, self.randomDim])
                imageBatch = self.X_train[np.random.randint(0, self.X_train.shape[0], size=batchSize)]

                # Generate fake images
                generatedImages = self.generator.predict(noise)
                X = np.concatenate([imageBatch, generatedImages])

                # Labels for generated and real data
                yDis = np.zeros(2*batchSize)
                # One-sided label smoothing
                yDis[:batchSize] = 0.9

                # Train discriminator
                self.discriminator.trainable = True
                dloss = self.discriminator.train_on_batch(X, yDis)

                # Train generator
                noise = np.random.normal(0, 1, size=[batchSize,self.randomDim])
                yGen = np.ones(batchSize)
                self.discriminator.trainable = False
                gloss = self.gan.train_on_batch(noise, yGen)

            # Store loss of most recent batch from this epoch
            print ("%d/%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (e,FLAGS.epochs, dloss, 100*dloss, gloss))

            self.dLosses.append(dloss)
            self.gLosses.append(gloss)

             # write tensorboard logs
            self.tensorboard.set_model(self.discriminator)
            self.tensorboard.on_epoch_end(e,{"loss":dloss , "accuracy":dloss})
            
            self.tensorboard.set_model(self.generator)
            self.tensorboard.on_epoch_end(e,{"loss":gloss , "accuracy":dloss})


            if e == 1 or e % FLAGS.save_step == 0:
                noise = np.random.normal(0, 1, size=[25, self.randomDim])
                imgs = self.generator.predict(noise)
                plot_generated_images(imgs,e)
                self.save_models(e)

        # Plot losses from every epoch
        #plot_loss(e , self.dLosses,self.gLosses)

    # Save the generator and discriminator networks (and weights) for later use
    def save_models(self,epoch):
        self.generator.save(f'{FLAGS.job_dir}/dcgan_generator_v2.h5')
        self.discriminator.save(f'{FLAGS.job_dir}/dcgan_discriminator_v2.h5')
        self.gan.save(f'{FLAGS.job_dir}/dcgan_combined_v2.h5')       


    def named_logs(self,model, logs):
        result = {}
        for l in zip(model.metrics_names, logs):
            result[l[0]] = l[1]
        return result     

if __name__ == '__main__':
    model = DCGAN_V2()
    model()