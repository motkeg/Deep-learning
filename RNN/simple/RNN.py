'''
we implement here a simle deep net with 3 hidden layers
'''

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.python.ops import rnn , rnn_cell

#init the data
MNIST = input_data.read_data_sets("/tmp/data/",one_hot=True)

# declear some size var



n_classes = 10 # nomber of clases to classify, 0-9

batch_size = 128
chunk_size = 28
n_chunks = 28
rnn_size = 128


X = tf.placeholder('float', [None, n_chunks , chunk_size])
Y = tf.placeholder('float')


def build_RNNnet_model(data):

#  init the output layer
   layer = {'weights': tf.Variable(tf.random_normal([rnn_size, n_classes])),
             'biases': tf.Variable(tf.random_normal([n_classes]))}

   data = tf.transpose(data,[1,0,2])
   data = tf.reshape(data,[-1, chunk_size])
   data = tf.split(data,n_chunks)

   lstm_cell = rnn_cell.BasicLSTMCell(rnn_size,state_is_tuple=True)
   outputs , states = rnn.static_rnn(lstm_cell , data , dtype=tf.float32)

   output  = tf.add(tf.matmul(outputs[-1] , layer['weights']) , layer['biases'])


   return output



def train_model(data ):

    prediction  = build_RNNnet_model(data)
    cost        = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=Y))
    optimizer   = tf.train.AdamOptimizer().minimize(cost)

    loops = 5

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for epoch in range(loops):
            epoch_loss = 0
            for _ in range(int(MNIST.train.num_examples / batch_size)):
                epoch_x, epoch_y = MNIST.train.next_batch(batch_size)
                epoch_x  = epoch_x.reshape(batch_size, n_chunks, chunk_size ) # we need to reshape the epoch_x output

                _, c = sess.run([optimizer, cost], feed_dict={X: epoch_x, Y: epoch_y})
                epoch_loss += c

            print('Epoch', epoch, 'completed out of', loops, ' | loss:', epoch_loss)

        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))  # like assertion - to verify our prediction is corespnod to label

        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('Accuracy:', accuracy.eval({X: MNIST.test.images.reshape(-1 ,n_chunks , chunk_size), Y: MNIST.test.labels}))



train_model(X)