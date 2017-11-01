import tflearn
from tflearn.layers.conv import conv_2d ,max_pool_2d
from tflearn.layers.core import input_data , dropout , fully_connected
from tflearn.layers.estimator import regression
import tflearn.datasets.mnist as MNIST


X, Y , test_X , test_Y = MNIST.load_data(one_hot=True)
X , test_X  = X.reshape([-1,28,28,1]) , test_X.reshape([-1,28,28,1])

CNN = input_data(shape=[None,28,28,1] ,name='input')
#layer 1 of conv
CNN = conv_2d(CNN , 32, 2 ,activation='relu')
CNN = max_pool_2d(CNN, 2)
#layer 2 of conv
CNN = conv_2d(CNN , 64, 2 ,activation='relu')
CNN = max_pool_2d(CNN, 2)

CNN = fully_connected(CNN , 1024 , activation='relu')
CNN = dropout(CNN ,0.85)

CNN = fully_connected(CNN , 10 ,activation="softmax")
CNN = regression(CNN, optimizer="adam" , learning_rate=0.01 , loss="categorical_crossentropy" , name="targets")

model = tflearn.DNN(CNN)

model.fit({'input':X},{"targets":Y} , n_epoch=10, validation_set=({'input':test_X},{"targets":test_Y}) ,
           snapshot_step=1000 , run_id="mnist" ,show_metric=True)

model.save("tflearn_CNN.model")

'''
model.load('tflearn_CNN.model')
import numpy as np
print("prediction: ", np.round(model.predict([test_X[1]])[0]) )
print("actual: " ,test_Y[1])'''






