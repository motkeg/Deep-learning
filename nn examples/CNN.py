'''
we implement here a simle deep net with 3 hidden layers
'''

import tensorflow as tf
import webbrowser
from tensorflow.examples.tutorials.mnist import input_data


#init the data
MNIST = input_data.read_data_sets("/tmp/data/",one_hot=True)

# declear some size var

n_classes = 10 # nomber of clases to classify, 0-9

batch_size = 128

X = tf.placeholder('float', [None, 784])
Y = tf.placeholder('float')

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def maxpool2d(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def build_CNNnet_model(data):

    #CNN  net with 2 layers
    weights = {
        # 5 x 5 convolution, 1 input image, 32 outputs
        'W_conv1': tf.Variable(tf.random_normal([3, 3, 1, 32])),
        # 5x5 conv, 32 inputs, 64 outputs
        'W_conv2': tf.Variable(tf.random_normal([3, 3, 32, 64])),
        # fully connected, 7*7*64 inputs, 1024 outputs
        'W_fc': tf.Variable(tf.random_normal([7 * 7 * 64, 1024])),
        # 1024 inputs, 10 outputs (class prediction)
        'out': tf.Variable(tf.random_normal([1024, n_classes]))
    }

    biases = {
        'b_conv1': tf.Variable(tf.random_normal([32])),
        'b_conv2': tf.Variable(tf.random_normal([64])),
        'b_fc': tf.Variable(tf.random_normal([1024])),
        'out': tf.Variable(tf.random_normal([n_classes]))
    }

    # Reshape input to a 4D tensor
    x = tf.reshape(data, shape=[-1, 28, 28, 1])
    # Convolution Layer, using our function
    conv1 = tf.nn.relu(conv2d(x, weights['W_conv1']) + biases['b_conv1'])
    # Max Pooling (down-sampling)
    conv1 = maxpool2d(conv1)
    # Convolution Layer
    conv2 = tf.nn.relu(conv2d(conv1, weights['W_conv2']) + biases['b_conv2'])
    # Max Pooling (down-sampling)
    conv2 = maxpool2d(conv2)

    # Fully connected layer
    # Reshape conv2 output to fit fully connected layer
    fc = tf.reshape(conv2, [-1, 7 * 7 * 64])
    fc = tf.nn.relu(tf.matmul(fc, weights['W_fc']) + biases['b_fc'])

    output = tf.matmul(fc, weights['out']) + biases['out']
    return output






def train_model(data ):

    prediction  = build_CNNnet_model(data)
    cost        = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=Y))
    optimizer   = tf.train.AdagradOptimizer(learning_rate=0.01).minimize(cost)

    loops = 5

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        tf.summary.FileWriter('./logs/CNN', graph=sess.graph)

        for epoch in range(loops):
            epoch_loss = 0
            for _ in range(int(MNIST.train.num_examples / batch_size)):
                epoch_x, epoch_y = MNIST.train.next_batch(batch_size)
                _, c = sess.run([optimizer, cost], feed_dict={X: epoch_x, Y: epoch_y})
                epoch_loss += c

            print('Iteration', epoch, 'completed out of', loops, ' | loss:', epoch_loss)

        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))  # like assertion - to verify our prediction is corespnod to label

        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('Accuracy:', accuracy.eval({X: MNIST.test.images, Y: MNIST.test.labels}))



train_model(X )