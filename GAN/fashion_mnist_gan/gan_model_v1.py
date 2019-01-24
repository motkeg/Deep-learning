import tensorflow as tf
from tensorflow import keras
flags = tf.app.flags
FLAGS = flags.FLAGS


USE_TPU = False

from tensorflow.keras.layers import (Input, Dense, Reshape, Flatten, Dropout,
                                    BatchNormalization, Activation, ZeroPadding2D,
                                    LeakyReLU, UpSampling2D, Conv2D)


from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.datasets import fashion_mnist as data

import numpy as np
import matplotlib.pyplot as plt
import os

'''if FLAGS.dataset == "fashion_mnist":
        from tensorflow.keras.datasets import fashion_mnist as data
    elif FLAGS.dataset == "mnist":
        from tensorflow.keras.datasets import mnist as data
    else:
        print ("ERROR:  wrong dataset name (try - [mnist,fashion-mnist])")'''

strategy = tf.contrib.tpu.TPUDistributionStrategy(
                        tf.contrib.cluster_resolver.TPUClusterResolver(tpu='grpc://' + os.environ['COLAB_TPU_ADDR'] ))         


class DCGAN():
    
    def __init__(self,train=True):
        
        
        self.img_shape = ( FLAGS.sample_size,  FLAGS.sample_size, FLAGS.c_dim)
        self.latent_dim = 100
        self.trained = train ## this variable use to determine if we want to train the model 

        self.optimizer = Adam(0.0002, 0.5)
        #define tensorboard
        self.tensorboard = keras.callbacks.TensorBoard(log_dir='./logs/gan',
                                                        batch_size=FLAGS.batch_size,
                                                        write_graph=True,
                                                        histogram_freq=0,
                                                        write_images=True,
                                                        write_grads=True)
        self.checkpointer  = keras.callbacks.ModelCheckpoint(filepath=f'{FLAGS.job_dir}/gan_model.best.hdf5', verbose = 1, save_best_only=True)                                                
        # Build the discriminator
        self.discriminator = self.build_D()
        self.discriminator.compile(loss='binary_crossentropy',
                                                optimizer=self.optimizer,
                                                metrics=['accuracy'])
        
        # Build the generator
        self.generator = self.build_G()

        # The generator takes noise as input and generates imgs
        z = Input(shape=(self.latent_dim,))
        img = self.generator(z)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The discriminator takes generated images as input and determines validity
        valid = self.discriminator(img)

        # The combined model  (stacked generator and discriminator)
        # Trains the generator to fool the discriminator
        self.combined = Model(z, valid)
        if USE_TPU:
           self.combined = tf.contrib.tpu.keras_to_tpu_model(self.combined, strategy=strategy)

        self.combined.compile(loss='binary_crossentropy', optimizer=self.optimizer)
        self.tensorboard.set_model(self.combined)
        


    
                                                                                                              


    def __call__(self):
        if self.trained:
            # Load the dataset
            (X_train, _), (_, _) = data.load_data()

            # Rescale -1 to 1
            X_train = X_train / 127.5 - 1.
            X_train = np.expand_dims(X_train, axis=3)
            #val_size = int(len(X_train)*0.1)
            #X_valid  = X_train[:val_size] 

            # Adversarial ground truths
            valid = np.ones((FLAGS.batch_size, 1))
            fake = np.zeros((FLAGS.batch_size, 1))

            for epoch in range(FLAGS.epoch):

                # ---------------------
                #  Train Discriminator
                # ---------------------

                # Select a random half of images
                idx = np.random.randint(0, X_train.shape[0], FLAGS.batch_size)
                imgs = X_train[idx]

                # Sample noise and generate a batch of new images
                noise = np.random.normal(0, 1, (FLAGS.batch_size, self.latent_dim))
                gen_imgs = self.generator.predict(noise)

                # Train the discriminator (real classified as ones and generated as zeros)
                self.discriminator.trainable = True   
                d_loss_real = self.discriminator.train_on_batch(imgs, valid)
                d_loss_fake = self.discriminator.train_on_batch(gen_imgs, fake)
               
                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

                #tf.summary.scalar("d_loss",d_loss) # add by moti
                

                # ---------------------
                #  Train Generator
                # ---------------------

                # Train the generator (wants discriminator to mistake images as real)
                self.discriminator.trainable = False
                g_loss = self.combined.train_on_batch(noise, valid)
                #tf.summary.scalar("g_loss",g_loss) # add by moti

                self.tensorboard.set_model(self.discriminator)
                self.tensorboard.on_epoch_end(epoch,self.named_logs(self.discriminator,d_loss_real))
                self.tensorboard.on_epoch_end(epoch,self.named_logs(self.discriminator,d_loss_fake))

                self.tensorboard.set_model(self.generator)
                self.tensorboard.on_epoch_end(epoch,self.named_logs(self.generator,g_loss))
                
               
                # Plot the progress
                print ("%d/%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch,FLAGS.epoch, d_loss[0], 100*d_loss[1], g_loss))
                self.tensorboard.on_epoch_end(epoch,{"loss":g_loss , "accuracy":d_loss[1]})
                self.combined.save("checkpoints/model_weights.h5")
                # If at save interval => save generated image samples
                if epoch % FLAGS.save_step == 0:
                    img  = self.save_imgs(epoch)
                    img_name = FLAGS.dataset +"_%d" % epoch
                    '''graph =  tf.keras.backend.get_session().graph
                    tf.summary.image(img_name,img)# add by moti
                    merge =  tf.summary.merge_all() # add by moti
                    G_sum = tf.summary.FileWriter('logs/G',graph) 
                    D_sum = tf.summary.FileWriter('logs/D',graph) 
                    #main_sum = tf.summary.FileWriter('logs/G',graph)     # add by moti  
                    D_sum.add_summary(self.discriminator.summary())
                    G_sum.add_summary(self.generator.summary())
                    D_sum.add_summary(self.combined.summary())'''
                    
                    

                
                   


    def named_logs(self,model, logs):
        result = {}
        for l in zip(model.metrics_names, logs):
            result[l[0]] = l[1]
        return result

    def build_G(self):
        with tf.variable_scope("generator", reuse=False):
            model = Sequential()

            model.add(Dense(128 * 7 * 7, activation="relu", input_dim=self.latent_dim))
            model.add(LeakyReLU(0.2))
            model.add(Reshape((7, 7, 128)))
            model.add(UpSampling2D(size=(2,2)))
            model.add(Conv2D(64, kernel_size=5, padding="same"))
            model.add(LeakyReLU(0.2))
            #model.add(BatchNormalization(momentum=0.8))
            #model.add(Activation("relu"))
            model.add(UpSampling2D(size=(2,2)))
            #model.add(Conv2D(1, kernel_size=5, padding="same" ,activation="tanh"))
            #model.add(BatchNormalization(momentum=0.8))
            #model.add(Activation("relu"))
            model.add(Conv2D(FLAGS.c_dim, kernel_size=5, padding="same",activation="tanh"))
            #model.add(Activation("tanh"))
            model.summary()

            model.compile(loss='binary_crossentropy', optimizer=self.optimizer)
            
            noise = Input(shape=(self.latent_dim,))
            img = model(noise)

            return Model(noise, img)

    def build_D(self):
         with tf.variable_scope("discriminator", reuse=False): 

            model = Sequential()

            model.add(Conv2D(32, kernel_size=5, strides=2, input_shape=self.img_shape, padding="same"))
            model.add(LeakyReLU(0.2))
            model.add(Dropout(0.3))
            model.add(Conv2D(128, kernel_size=5, strides=2, padding="same"))
            #model.add(ZeroPadding2D(padding=((0,1),(0,1))))
            #model.add(BatchNormalization(momentum=0.8))
            model.add(LeakyReLU(alpha=0.2))
            model.add(Dropout(0.3))
            '''model.add(Conv2D(128, kernel_size=3, strides=2, padding="same"))
            model.add(BatchNormalization(momentum=0.8))
            model.add(LeakyReLU(alpha=0.2))
            model.add(Dropout(0.25))
            model.add(Conv2D(256, kernel_size=3, strides=1, padding="same"))
            model.add(BatchNormalization(momentum=0.8))
            model.add(LeakyReLU(alpha=0.2))
            model.add(Dropout(0.25))'''
            model.add(Flatten())
            model.add(Dense(1, activation='sigmoid'))

            model.summary()
            model.compile(loss='binary_crossentropy', optimizer=self.optimizer)

            img = Input(shape=self.img_shape)
            validity = model(img)

            return Model(img, validity)


    def save_imgs(self, epoch):
        r, c = 5, 5
        noise = np.random.normal(0, 1, (r * c, self.latent_dim))
        gen_imgs = self.generator.predict(noise)

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs[cnt, :,:,0], cmap='gray')
                axs[i,j].axis('off')
                cnt += 1
        name = FLAGS.dataset + "_{}.png".format(epoch)        
        fig.savefig("samples/"+FLAGS.dataset +"/" + name)
        plt.close()
        return gen_imgs
                                         