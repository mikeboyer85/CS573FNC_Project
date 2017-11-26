###################################################
# FNC-1 Neural Network Implementation
# Mike Boyer 
# 2017/11/06 
#
# 2017/11/12    Mike Boyer    fnc1_cnn design / framework
###################################################
import numpy as np
import tensorflow as tf
from Vectors import *
from utils import *
from datetime import datetime

trainStancesFile = './data/train_stances_related_only.csv'
trainBodiesFile = './data/train_bodies.csv'
testStancesFile = './data/test_stances_unlabeled.csv'
testBodiesFile = './data/test_bodies.csv'

#Static Params for FNC-1 / CS573 version
stances = {'agree':0,'disagree':1,'discuss':2}
n_classes = 3
n_hidden = 256      #256 nodes in the first hidden layer. Other layers are multiples of this
n_embDims = 300     #300 embedded dimensions considered in the NN.
n_trainBatchSize = 32     #Train on 32 stances at a time
n_trainSteps = 10  #Solat used 35,000,000 
n_dense = 1024

#The CNN will have 4 layers. hidden size increase with depth (ratios copied from SOLAT team)
weights = {
    'c1': tf.Variable(tf.random_normal([n_embDims, n_hidden])),
    'c2': tf.Variable(tf.random_normal([n_hidden, n_hidden])),
    'c3': tf.Variable(tf.random_normal([n_hidden, n_hidden*2])),
    'c4': tf.Variable(tf.random_normal([n_hidden*2, n_hidden*3])), 
    'c5': tf.Variable(tf.random_normal([n_hidden*3, n_classes])), 
    }

biases = {
    'c1': tf.Variable(tf.random_normal([n_hidden])),
    'c2': tf.Variable(tf.random_normal([n_hidden])),
    'c3': tf.Variable(tf.random_normal([n_hidden*2])),
    'c4': tf.Variable(tf.random_normal([n_hidden*3])),
    'c5': tf.Variable(tf.random_normal([n_classes]))
    }

AdamParams = {
     'learning_rate': 0.001,
     'beta1': 0.1,
     'beta2': 0.001,
     'e': 1e-8
     } 

def conv_net (x_tensor, is_training, reuse):
    #Convolution Layer with 300 input dims, 256 output dims, width 3, pad 3, stride 1, and relu activation
    conv1 = tf.layers.conv2d(x_tensor, filters=n_hidden, kernel_size = 3, strides = 1, padding = 'same', activation = tf.nn.relu)
    #conv1 = tf.layers.conv1d(x_tensor, filters=n_hidden, kernel_size=3, strides=1, padding='same', activation = tf.nn.relu)
    #Apply dropout
    #conv1 = tf.layers.dropout(conv1, rate=0.5, training=is_training)
    #Pooling layer with stride (1,2) and pad (1,2)
    #conv1 = tf.layers.max_pooling1d(inputs = conv1, pool_size = 2, strides = 2, padding='same')
    
    #Convolution layer with 256 input dims, 256 output dims, width 3, pad 3, stride 1, and relu activation
    #conv2 = tf.layers.conv1d(conv1, filters=n_hidden, kernel_size=3, strides=1, padding='same', activation = tf.nn.relu)
    #conv2 = tf.layers.dropout(conv2, rate=0.5, training=is_training)
    #conv2 = tf.layers.max_pooling1d(inputs=conv2, pool_size = 2, strides = 2, padding = 'same')
    
    #Convolution layer with 256 input dims, 512 output dims, width 3, pad 3, stride 1, and relu activation
    #conv3 = tf.layers.conv1d(conv2, filters=n_hidden*2, kernel_size=3, strides=1, padding='same', activation = tf.nn.relu)
    #conv3 = tf.layers.dropout(conv3, rate=0.5, training=is_training)
    #conv3 = tf.layers.max_pooling1d(inputs=conv3, pool_size=2, strides=2, padding='same')
    
    #Convolution layer with 512 input dims, 512 output dims, width 3, pad 3, stride 1, and relu activation
    #conv4 = tf.layers.conv1d(conv3, filters=n_hidden*2, kernel_size=3, strides=1, padding='same', activation = tf.nn.relu)
    #conv4 = tf.layers.dropout(conv4, rate=0.5, training=is_training)
    #conv4 = tf.layers.max_pooling1d(inputs=conv4, pool_size=2, strides=2, padding='same')
    
    #Convolution layer with 512 input dims, 768 output dims, width 3, pad 1, stride 1, and relu activation
    #conv5 = tf.layers.conv1d(conv4, filters=n_hidden*3, kernel_size=3, strides=1, padding='same', activation = tf.nn.relu)
    #conv5 = tf.layers.dropout(conv5, rate=0.5, training=is_training)
    #conv5 = tf.layers.max_pooling1d(inputs=conv5, pool_size=2, strides=2, padding='same')
    return conv1

def full_net(h_tensor, b_tensor, is_training):
    #A fully connected multi-layer perceptron. Final layer is softmax
    
    h_conv = conv_net(h_tensor, is_training=is_training, reuse=False)
    b_conv = conv_net(b_tensor, is_training=is_training, reuse=False)
    
    #TODO - check this.. is it actually right?
    feature_vec = tf.concat(h_conv, b_conv, axis=1)

    d1 = tf.layers.dense(feature_vec, 1024, activation=tf.nn.relu)
    d2 = tf.layers.dense(d1, 1024, activation = tf.nn.relu)
    d3 = tf.layers.dense(d2, 1024, activation = tf.nn.relu)
    out = tf.layers.dense(d3, n_classes, activation = tf.nn.softmax)
    return out


def train(trainData):
    ############################################
    # Assumptions:
    # 1. Label is always the last column in the stancesFileName
    ############################################
    self.trainStart = datetime.now()
    print('Training Start: {:%H:%M:%S} ... Total Elapsed Time:{}'.format(self.trainStart ,(self.trainStart - self.theBeginning)))
    
    heads, bodies, stances = trainData.sample(n = trainData.n_stances)
    feature_vec_idxs = np.concatenate((heads, bodies),axis=1)   #concatenate the heads and bodies
    self.feature_vec = self.gv.model.wv[feature_vec_idxs]       #shape is n x 300
    #num_dims = feature_vec.shape[0]        
    
    #Placeholders for data / labels
    X = tf.placeholder('float', [None, n_embDims])      # - X - the training data
    Y = tf.placeholder('float', [None, n_classes])    # - Y - the training labels
    
    #Build the neural network in TF variables
    logits = cnn.neural_net(X)
    
    #define loss and optimizer
    loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels = Y))
    optimizer = tf.train.AdamOptimizer(learning_rate = AdamParams['learning_rate'],
                                       beta1 = AdamParams['beta1'],
                                       beta2 = AdamParams['beta2'],
                                       epsilon = AdamParams['e'])
    train_op = optimizer.minimize(loss_op)
    
    #Evaluate model (with test logits, for dropout to be disabled)
    correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    
    #Initialize the variables (i.e. assign their default value)
    init = tf.global_variables_initializer()
    
    #Start Training
    with tf.Session() as sess:
        #Run initializer
        sess.run(init)
        
        #for step in range(1, n_trainSteps + 1):
            #raise Exception('Start here tomorrow, mike !!!')
            #TODO
            #batch_x, batch_y = self.getNextTrainData(self.feature_vec, stances, n_trainBatchSize)
#                batch_x, batch_y = trainData.sample(n = n_trainBatchSize)
#                #batch_x, batch_y = mnist.train.next_batch(batch_size)
#                #run optimization op (backprop)
#                sess.run(train_op, feed_dict = {X: batch_x, Y:batch_y})
#                if step % display_step == 0 or step == 1:
#                    #calculate loss and accuracy
#                    loss, acc = sess.run([loss_op, accuracy], feed_dict = {X: batch_x, Y: batch_y})
#                    print("Step " + str(step) + ", Minibatch Loss= " + \
#                          "{:.4f}".format(loss) + ", Training Accuracy= " + \
#                          "{:.3f}".format(acc))
#            
#            print('Optimization Finished!!')
#            
#            # Calculate accuracy for MNIST test images
#            print("Testing Accuracy:", \
#                sess.run(accuracy, feed_dict={X: mnist.test.images,
#                                              Y: mnist.test.labels}))

    self.trainEnd = datetime.now()
    print('Training End: {:%H:%M:%S} ... Total Elapsed Time:{}'.format(self.trainEnd ,(self.trainEnd - self.theBeginning)))   
    
def predict(self, bodies):
    raise Exception('Predicition is not implemented yet!!!!')

def validate(self, correctStances, predStances):
    raise Exception('validation is not implemented yet!!!!!')

    
##########################################
# Step 1 - Generate features for the data
##########################################
theBeginning = datetime.now()
print('Program start: {:%H:%M:%S}'.format(theBeginning))
v = GoogleVec()
v.load()
emb_mat = v.model.syn0

trainData = fnc1_Data(stancesFile=trainStancesFile, bodiesFile=trainBodiesFile, vecs=v)

##########################################
# Step 2 - A simple training loop
##########################################
with tf.Session() as sess:
    #init = tf.global_variables_initializer()
    init = tf.initialize_all_variables()
    sess.run(init)
    
    #TODO - for loop for iterating over batches
    h_idxs, b_idxs, labels = trainData.sample(n=1, ridx=range(batch_size))
    h_vecs = emb_mat[h_idxs]
    b_vecs = emb_mat[b_idxs]
    
    #TEST
    out = conv_net(h_tensor, is_training=True, reuse=False)
    print(out.shape)
    
    
