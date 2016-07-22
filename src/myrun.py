from network3 import *
mini_batch_size = 10
import theano.misc.pkl_utils
with open('/mnt/data.zip','rb') as f:
  training_data, validation_data, test_data = theano.misc.pkl_utils.load(f)
#training_data, validation_data, test_data = network3.load_all_data('/mnt/captchas')
net = Network([                                 
        ConvPoolLayer(image_shape=(mini_batch_size, 1, 70, 200),
                      filter_shape=(20, 1, 5, 5),
                      poolsize=(2, 2)),
        FullyConnectedLayer(n_in=20*(200-4)/2*(70-4)/2, n_out=100),
        SoftmaxLayer(n_in=100, n_out=26)], mini_batch_size)
net.SGD(training_data, 60, mini_batch_size, 0.1, 
            validation_data, test_data)
