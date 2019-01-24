
import tensorflow as tf
import os , time
from data import *
#from sklearn.utils import shuffle

# Turn off TensorFlow warning messages in program output
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

train ,test, eval = get_data() #- data  proccess from data.py

'''
we implement here a simle deep net with 3 hidden layers
'''

# Define model parameters
learning_rate = 0.001
training_epochs = 20

# Define how many inputs and outputs are in our neural network
number_of_inputs = 30 # 30 colums in our data file
number_of_outputs = 1

# Define how many neurons we want in each layer of our neural network
layer_1_nodes = 50
layer_2_nodes = 100
layer_3_nodes = 50




''' build the network '''

#Input layer
with tf.variable_scope("input_layer"):
    X = tf.placeholder(dtype=tf.float64 , name="input" , shape=(None,number_of_inputs))

#Layer 1
with tf.variable_scope("layer1"):
    weights = tf.get_variable(dtype=tf.float64,name="weights1", shape=[number_of_inputs, layer_1_nodes],initializer=tf.contrib.layers.xavier_initializer())
    biases = tf.get_variable(dtype=tf.float64,name="biases1", shape=[layer_1_nodes], initializer=tf.zeros_initializer())
    layer_1_output = tf.nn.relu(tf.matmul(X, weights) + biases)

#Layer 2
with tf.variable_scope("layer2"):
    weights = tf.get_variable(dtype=tf.float64,name="weights2", shape=[layer_1_nodes, layer_2_nodes],initializer=tf.contrib.layers.xavier_initializer())
    biases = tf.get_variable(dtype=tf.float64,name="biases2", shape=[layer_2_nodes], initializer=tf.zeros_initializer())
    layer_2_output = tf.nn.relu(tf.matmul(layer_1_output, weights) + biases)

#Layer 3
with tf.variable_scope("layer3"):
    weights = tf.get_variable(dtype=tf.float64,name="weights3", shape=[layer_2_nodes, layer_3_nodes],initializer=tf.contrib.layers.xavier_initializer())
    biases = tf.get_variable(dtype=tf.float64,name="biases3", shape=[layer_3_nodes], initializer=tf.zeros_initializer())
    layer_3_output = tf.nn.relu(tf.matmul(layer_2_output, weights) + biases)

#Output Layer
with tf.variable_scope("output_layer"):
    weights = tf.get_variable(dtype=tf.float64,name="out_weights", shape=[layer_3_nodes, number_of_outputs],initializer=tf.contrib.layers.xavier_initializer())
    biases = tf.get_variable(dtype=tf.float64,name="out_biases", shape=[number_of_outputs], initializer=tf.zeros_initializer())
    prediction = tf.nn.relu(tf.matmul(layer_3_output, weights) + biases)

    # add this opration to the graph that we can run it after
    tf.add_to_collection("predict" ,prediction)

''' define cost and optimazer'''

with tf.variable_scope("cost"):
    Y = tf.placeholder(dtype=tf.float64, shape=(None, 1) ,name="prediction")
    cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=prediction, labels=Y))
    # add this opration to the graph that we can run it after
    tf.add_to_collection("cost", cost)

with tf.variable_scope("optimazer"):
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
    # add this opration to the graph that we can run it after
    tf.add_to_collection("optimize", optimizer)

with tf.variable_scope('logging'):
    tf.summary.scalar('current_cost', cost)
    tf.summary.histogram('predicted_value', prediction)
    summary = tf.summary.merge_all()


saver = tf.train.Saver()
save_path = "models/" + time.strftime("%Y%m%d%H%M%S")
#run the nurural net
with tf.Session() as sess:
    # Run the global variable initializer to initialize all variables and layers of the neural network
    sess.run(tf.global_variables_initializer())

    # Create log file writers to record training progress.
    # We'll store training and testing log data separately.
    training_writer = tf.summary.FileWriter('./logs/training', sess.graph)
    testing_writer = tf.summary.FileWriter('./logs/testing', sess.graph)

    # run the net and the optimazer
    for epoch in range(training_epochs):
        sess.run(optimizer , feed_dict={X:train["X"],Y:train["Y"]})

        #print some data too see the progress
        if epoch % 2 == 0:
            # Get the current accuracy scores by running the "cost" operation on the training and test data sets
            training_cost, training_summary = sess.run([cost, summary],
                                                          feed_dict={X:train["X"],Y:train["Y"]})
            testing_cost, testing_summary = sess.run([cost, summary],
                                                        feed_dict={X:test["X"],Y:test["Y"]})

            # Write the current training status to the log files (Which we can view with TensorBoard)
            training_writer.add_summary(training_summary, epoch)
            testing_writer.add_summary(testing_summary, epoch)
            # like assertion - to verify our prediction is corespnod to label
            correct = tf.equal(tf.argmax(prediction, 1),tf.argmax(Y, 1))
            accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
            print("Epoch: {} - Training Cost: {}  Testing Cost: {} ".format(epoch, training_cost, testing_cost))
            #print('Accuracy:', accuracy.eval({X: X_testing, Y: Y_testing}))

            # Training is now complete!

    # Get the final accuracy scores by running the "cost" operation on the training and test data sets
    final_training_cost = sess.run(cost, feed_dict={X:train["X"],Y:train["Y"]})
    final_eval_cost = sess.run(cost, feed_dict={X:eval["X"],Y:eval["Y"]})

    saver.save(sess, save_path="./models/simple_nn")
    print("Final Training cost: {}".format(final_training_cost))
    print("Final evaluate cost: {}".format(final_eval_cost))



