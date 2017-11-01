'''
we implement here a simle deep net with 3 hidden layers
'''

import tensorflow as tf
import  webbrowser , subprocess
from tensorflow.examples.tutorials.mnist import input_data

#init the data
MNIST = input_data.read_data_sets("/tmp/data/",one_hot=True)

# declear some size var

n_classes = 10 # nomber of clases to classify, 0-9

batch_size = 128

X = tf.placeholder('float', [None, 784])
Y = tf.placeholder('float')


def build_net_model(data , n_layers , nodes_len_Arr):
    LAYERS = {}

   #  init the first layer
    LAYERS[0] = {'weights': tf.Variable(tf.random_normal([784, nodes_len_Arr[0]])),
                        'biases': tf.Variable(tf.random_normal([nodes_len_Arr[0]]))}
    LAYERS[0] = tf.nn.relu_layer(data,LAYERS[0]['weights'],LAYERS[0]['biases'])


    for i in range(1,n_layers):
    #  init the middle layers
        LAYERS[i] = {'weights':tf.Variable(tf.random_normal([nodes_len_Arr[i-1], nodes_len_Arr[i]])),
                          'biases':tf.Variable(tf.random_normal([nodes_len_Arr[i]]))}
        LAYERS[i] = tf.nn.relu_layer(LAYERS[i-1],LAYERS[i]['weights'],LAYERS[i]['biases'])



    #  init the output layer
    LAYERS["out"] = {'weights': tf.Variable(tf.random_normal([nodes_len_Arr[n_layers-1], n_classes])),
                     'biases': tf.Variable(tf.random_normal([n_classes]))}

    LAYERS["out"] = tf.nn.relu_layer(LAYERS[n_layers-1], LAYERS["out"]['weights'],LAYERS["out"]['biases'])
    #print("L:", LAYERS)
    return LAYERS["out"]



def train_model(data , n_layers , nodes_len_Arr):

    prediction  = build_net_model(data, n_layers , nodes_len_Arr)
    cost        = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=Y))
    optimizer   = tf.train.AdagradOptimizer(learning_rate=0.01).minimize(cost)

    loops = 10

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        tf.summary.FileWriter('./logs/', graph=sess.graph)
        for epoch in range(loops):
            epoch_loss = 0

            for _ in range(int(MNIST.train.num_examples / batch_size)):
                epoch_x, epoch_y = MNIST.train.next_batch(batch_size)
                _, c = sess.run([optimizer, cost], feed_dict={X: epoch_x, Y: epoch_y})
                epoch_loss += c

            print('Epoch', epoch, 'completed out of', loops, ' | loss:', epoch_loss)

        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))  # like assertion - to verify our prediction is corespnod to label

        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('Accuracy:', accuracy.eval({X: MNIST.test.images, Y: MNIST.test.labels}))
        #subprocess.call(["tensorboard" , "--logdir" , "logs"])
        #webbrowser.open("http://localhost:6006", new=2)




train_model(X ,4,[512,512,512,512])
