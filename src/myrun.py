from network3 import *
mini_batch_size = 10
import theano.misc.pkl_utils
with open('/mnt/data.zip','rb') as f:
  training_data, validation_data, test_data = theano.misc.pkl_utils.load(f)
#training_data, validation_data, test_data = network3.load_all_data('/mnt/captchas')
#with open('/mnt/data.zip', 'wb') as f:
#  theano.misc.pkl_utils.dump((training_data, validation_data, test_data),f)
net = Network([                                 
        ConvPoolLayer(image_shape=(mini_batch_size, 1, 70, 200),
                      filter_shape=(20, 1, 5, 5),
                      poolsize=(2, 2)),
        FullyConnectedLayer(n_in=20*(200-4)/2*(70-4)/2, n_out=100),
        SoftmaxLayer(n_in=100, n_out=26)], mini_batch_size)
net.SGD(training_data, 60, mini_batch_size, 0.1, 
            validation_data, test_data)


#try 2
net = Network([                                   
        ConvPoolLayer(image_shape=(mini_batch_size, 1, 70, 200),
                      filter_shape=(40, 1, 7, 5),
                      poolsize=(2, 2)),
        ConvPoolLayer(image_shape=(mini_batch_size, 40, (70-6)/2, (200-4)/2),
                      filter_shape=(40, 40, 5, 5),
                      poolsize=(2, 2)),
        FullyConnectedLayer(n_in=40 * ((200-4)/2-4)/2 * ((70-6)/2-4)/2, n_out=100),
        SoftmaxLayer(n_in=100, n_out=26)], mini_batch_size)

net.SGD(training_data, 200, mini_batch_size, 0.02,
            validation_data, test_data)
            
#try 3 halved the size of the input data
net = Network([                                   
        ConvPoolLayer(image_shape=(mini_batch_size, 1, 36, 100),
                      filter_shape=(40, 1, 5, 5),
                      poolsize=(2, 2)),
        ConvPoolLayer(image_shape=(mini_batch_size, 40, (36-4)/2, (100-4)/2),
                      filter_shape=(40, 40, 5, 5),
                      poolsize=(2, 2)),
        FullyConnectedLayer(n_in=40 * ((36-4)/2-4)/2 * ((100-4)/2-4)/2, n_out=100),
        SoftmaxLayer(n_in=100, n_out=26)], mini_batch_size)

net.SGD(training_data, 200, mini_batch_size, 0.02,
            validation_data, test_data)

#try 4 set both ConvPoolLayer to have 60 instead of 40 layers.. source not here

#try 5 # using sing rectified linear units & l2 regularization
from network3 import ReLU
net = Network([                                   
        ConvPoolLayer(image_shape=(mini_batch_size, 1, 36, 100),
                      filter_shape=(40, 1, 5, 5),
                      poolsize=(2, 2), activation_fn=ReLU),
        ConvPoolLayer(image_shape=(mini_batch_size, 40, (36-4)/2, (100-4)/2),
                      filter_shape=(40, 40, 5, 5),
                      poolsize=(2, 2), activation_fn=ReLU),
        FullyConnectedLayer(n_in=40 * ((36-4)/2-4)/2 * ((100-4)/2-4)/2, n_out=100, activation_fn=ReLU),
        SoftmaxLayer(n_in=100, n_out=26)], mini_batch_size)

net.SGD(training_data, 200, mini_batch_size, 0.02,
            validation_data, test_data, lmbda=0.1) #might remove lmbda if it does work well

#try 6 set both ConvPoolLayer to have 80 instead of 40 layers
net = Network([                                   
        ConvPoolLayer(image_shape=(mini_batch_size, 1, 36, 100),
                      filter_shape=(80, 1, 5, 5),
                      poolsize=(2, 2)),
        ConvPoolLayer(image_shape=(mini_batch_size, 80, (36-4)/2, (100-4)/2),
                      filter_shape=(80, 80, 5, 5),
                      poolsize=(2, 2)),
        FullyConnectedLayer(n_in=80 * ((36-4)/2-4)/2 * ((100-4)/2-4)/2, n_out=100),
        SoftmaxLayer(n_in=100, n_out=26)], mini_batch_size)

net.SGD(training_data, 200, mini_batch_size, 0.02, validation_data, test_data)

#try 7 two fully connected layers 1000 nodes with drop out
net = Network([                                   
        ConvPoolLayer(image_shape=(mini_batch_size, 1, 36, 100),
                      filter_shape=(80, 1, 5, 5),
                      poolsize=(2, 2)),
        ConvPoolLayer(image_shape=(mini_batch_size, 80, (36-4)/2, (100-4)/2),
                      filter_shape=(80, 80, 5, 5),
                      poolsize=(2, 2)),
        FullyConnectedLayer(n_in=80 * ((36-4)/2-4)/2 * ((100-4)/2-4)/2, n_out=1000, p_dropout=0.5),
	FullyConnectedLayer(n_in=1000, n_out=1000, p_dropout=0.5),
        SoftmaxLayer(n_in=1000, n_out=26, p_dropout=0.5)
     ], mini_batch_size)
  
 net.SGD(training_data, 200, mini_batch_size, 0.02, validation_data, test_data)
