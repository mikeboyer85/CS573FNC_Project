###################################################
# FNC-1 Neural Network Implementation
# Mike Boyer 
# 2017/11/06 
#
# 2017/11/22   Mike Boyer    simplify / rewrite again
###################################################
import numpy as np
import tensorflow as tf
import time
from datetime import datetime
from Vectors import *
from utils import *

#Mode - 'train' or 'pred'
mode = 'train' # train or pred

#DataFiles
trainStancesFile = './data/train_stances_related_only.csv'
trainBodiesFile = './data/train_bodies.csv'
testStancesFile = './data/comp_test_stances_related_only.csv'
testBodiesFile = './data/competition_test_bodies.csv'

#Static Params for FNC-1 / CS573 version
classes = {'agree':0,'disagree':1,'discuss':2}
n_classes = 3       #agree:0, disagree:1, discuss:2
n_hidden = 32      #256 nodes in the first hidden layer. Other layers are multiples of this
n_embDims = 300     #300 embedded dimensions considered in the NN.
batch_size = 3     #Train on n stances at a time
n_steps = 20   #training steps. Solat used 35,000,000 
keep_prob = 0.5 #for dropout = the probability that a node will be kept.
learning_rate = 0.001

#Tensorflow Placeholders - These are all populated with values at runtime
h_tensor = tf.placeholder(tf.float32, shape=(None, None, 300, None))        #Tensor for headlines
b_tensor = tf.placeholder(tf.float32,  shape=(None, None, 300, None))        #Tensor for bodies
label_tensor = tf.placeholder(tf.int64, shape=(None, n_classes))      #Tensor for labels
feature_vec = tf.placeholder(tf.float32, shape=(1, n_hidden*6))

#TODO i want to get rid of these by using Layers library.
weights = {
    'c1': tf.Variable(tf.random_normal([5, 5, 1,            n_hidden   ]), name='wc1'),
    'c2': tf.Variable(tf.random_normal([5, 5, n_hidden,     n_hidden   ]), name='wc2'),
    'c3': tf.Variable(tf.random_normal([5, 5, n_hidden,     n_hidden*2 ]), name='wc3'),
    'c4': tf.Variable(tf.random_normal([5, 5, n_hidden*2,   n_hidden*2 ]), name='wc4'), 
    'c5': tf.Variable(tf.random_normal([5, 5, n_hidden*2,   n_hidden*3 ]), name='wc5'),
    'd1': tf.Variable(tf.random_normal([n_hidden*6, 1024]), name='wd1'),
    'd2': tf.Variable(tf.random_normal([1024, 1024]), name='wd2'),
    'd3': tf.Variable(tf.random_normal([1024, 1024]), name='wd3'),
    'd4': tf.Variable(tf.random_normal([1024, n_classes]), name='wd4')
    }

biases = {
    'c1': tf.Variable(tf.random_normal([n_hidden]), name='bc1'),
    'c2': tf.Variable(tf.random_normal([n_hidden]), name='bc2'),
    'c3': tf.Variable(tf.random_normal([n_hidden*2]), name='bc3'),
    'c4': tf.Variable(tf.random_normal([n_hidden*2]), name='bc4'),
    'c5': tf.Variable(tf.random_normal([n_hidden*3]), name='bc5'),
    'd1': tf.Variable(tf.random_normal([1024]), name='bd1'),
    'd2': tf.Variable(tf.random_normal([1024]), name='bd2'),
    'd3': tf.Variable(tf.random_normal([1024]), name='bd3'),
    'd4': tf.Variable(tf.random_normal([n_classes]), name='bd4')
    }


# Create some wrappers for simplicity
def conv2d(x, W, b, strides=1, keep_prob=0.5, reuse=True):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME', use_cudnn_on_gpu=False)
    x = tf.nn.bias_add(x, b)
    x = tf.nn.relu(x)
    if keep_prob != None:
        x=tf.nn.dropout(x, keep_prob=keep_prob)
    return x


def maxpool2d(x, k=2):
    # MaxPool2D wrapper
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                          padding='SAME')

def conv_net(x, weights, biases):   
    #c1 - 2d convolution layer, filter = [5,5,1,256] ->256 'features' to learn
    conv1 = conv2d(x, weights['c1'], biases['c1'])  
    conv1 = maxpool2d(conv1, k=2)
        
    #c2 - 2d convolution layer, filter = [5,5,256,256] ->256 'features' to re-learn
    conv2 = conv2d(conv1, weights['c2'], biases['c2'])
    conv2 = maxpool2d(conv2, k=2)
    
    #c3 - 2d convolution layer, filter = [5,5,256,512]
    conv3 = conv2d(conv2, weights['c3'], biases['c3'])
    conv3 = maxpool2d(conv3, k=2)
    
    #c4 - 2d convolution layer, filter = [5,5,512,512]
    conv4 = conv2d(conv3, weights['c4'], biases['c4'])
    
    #c5 - 2d convolution layer, filter = [5,5,512,768]
    conv5 = conv2d(conv4, weights['c5'], biases['c5'])

    return conv5

def full_net(hTens, bTens, weights, biases):
    #Calc conv for head and body. Drop initial dims and keep the learned features
    head_conv = tf.reshape(tf.reduce_max(conv_net(hTens, weights, biases), axis=[1,2]), [-1, n_hidden*3])
    body_conv = tf.reshape(tf.reduce_max(conv_net(bTens, weights, biases), axis=[1,2]), [-1, n_hidden*3]) 
    
    feature_vec = tf.reshape(tf.stack([head_conv, body_conv]),[-1, n_hidden*6])
    
    # Fully connected layer
    # Reshape conv2 output to fit fully connected layer input
    d1 = tf.add(tf.matmul(feature_vec, weights['d1']), biases['d1'])
    d1 = tf.nn.relu(d1)   

    d2 = tf.add(tf.matmul(d1, weights['d2']), biases['d2'])
    d2 = tf.nn.relu(d2)
    
    d3 = tf.add(tf.matmul(d2, weights['d3']), biases['d3'])
    d3 = tf.nn.relu(d3)
    
    d4 = tf.add(tf.matmul(d3, weights['d4']), biases['d4'])
    d4 = tf.nn.relu(d4)
    
    return d4

#Diag code. Delete later
def printGlobals():
    return len(tf.global_variables())
def printLocals():
    return len(tf.local_variables())
def printTrainables():
    return len(tf.trainable_variables())


##########################################
# Step 0 - Construct the model
##########################################
#head_conv = conv_net(h_tensor)
#body_conv = conv_net(b_tensor)   
logits = full_net(h_tensor, b_tensor, weights, biases)       #logits - the output of our neural network
preds = tf.nn.softmax(logits)               #preds - the prediction for the stance (0-1 for each class)

# Define loss and optimizer
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        logits=logits, labels=label_tensor))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.1, beta2=0.001, epsilon=1e-8)
train_op = optimizer.minimize(loss_op)

# Evaluate model
correct_pred = tf.equal(tf.argmax(preds, 1), tf.argmax(label_tensor, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


##########################################
# Step 1 - Generate features for the data
##########################################
theBeginning = datetime.now()
t00 = time.time()
print('Program start: {:%H:%M:%S}'.format(theBeginning))
v = GoogleVec()
print('Loading GoogleVectors...')
t0 = time.time()
v.load()
print('Loaded. ET: {}'.format(time.time() - t0))

init = tf.global_variables_initializer()
##########################################
# Step 2 - A simple training loop
##########################################
t0 = time.time()

with tf.Session() as sess:
    print('Initializing Session...')
    #Initialize the variables
    sess.run([init])
    #Init our saver (for saving and reading vars)
#    saver = tf.train.Saver({'wc1':weights['c1'],'wc2':weights['c2'], 'wc3':weights['c3'], 'wc4':weights['c4'],
#                    'wc5':weights['c5'], 'wd1':weights['d1'], 'wd2':weights['d2'], 'wd3':weights['d3'],
#                    'wd4':weights['d4'],
#                    'bc1':biases['c1'], 'bc2':biases['c2'], 'bc3':biases['c3'], 'bc4':biases['c4'],
#                    'bc5':biases['c5'], 'bd1':biases['d1'], 'bd2':biases['d2'], 'bd3':biases['d3'],
#                    'bd4':biases['d4']}, filename='./fnc1_params.ckpt')
    saver = tf.train.Saver(filename='./fnc1_params.ckpt')
    
    print('Session Initialized. ET:{}'.format(time.time() - t0))
    t0 = time.time()
    if mode == 'pred':
        #Restored saved params from previous training
        saver.restore(sess, './fnc1_trained_params.ckpt')
        
        #Load test data
        print('Loading test data')
        #TODO - load test data
        testData = fnc1_Data(stancesFile = testStancesFile, bodiesFile=testBodiesFile, vecs=v)
        #outFile = './output.csv'
        #with open(outFile,'w') as f:
        #    writer = csv.writer(f)
        #    writer.writerow(['Headline','BodyID','Agree', 'Disagree','Discuss','Unrelated'])
          
        #TODO - save predictions to file
        for i in range(testData.n_stances):
            h_idxs, b_idxs, labels = testData.getNextTrainBatch(1)
            h_vecs = np.expand_dims(v.model.syn0[h_idxs], axis=3)
            b_vecs = np.expand_dims(v.model.syn0[b_idxs], axis=3)
            
            p = sess.run([preds], feed_dict={h_tensor:h_vecs, 
                         b_tensor:b_vecs})
            print(p)

        #TODO - output confusion matrix
        
    elif mode == 'train':
        print('Loading training data...')
        trainData = fnc1_Data(stancesFile=trainStancesFile, bodiesFile=trainBodiesFile, vecs=v)

        #h_idxs, b_idxs, labels = trainData.getNextTrainBatch(batch_size)
        h_idxs, b_idxs, labels = trainData.getNextTrainBatch(batch_size)
        h_vecs = np.expand_dims(v.model.syn0[h_idxs], axis=3)
        b_vecs = np.expand_dims(v.model.syn0[b_idxs], axis=3)

        #Training loop - Train over all of our training data in each batch (mini-batch gradient desc)
        for step in range(n_steps):
            t1 = time.time()
            #h_idxs, b_idxs, labels = trainData.getNextTrainBatch(batch_size)
            #h_vecs = np.expand_dims(v.model.syn0[h_idxs], axis=3)
            #b_vecs = np.expand_dims(v.model.syn0[b_idxs], axis=3)
            print(h_vecs.shape, b_vecs.shape)
            sess.run([train_op], feed_dict={h_tensor:h_vecs, 
                                 b_tensor:b_vecs, label_tensor:labels})
            #print('Step: {}, Time: {}'.format(step, time.time()-t1))
            
            #if step % 500 == 0 or step == 0:
                #Print some nice debug info and persist our best info
            loss, acc, p = sess.run([loss_op, accuracy, preds], feed_dict={h_tensor:h_vecs, 
                                 b_tensor:b_vecs, label_tensor:labels})
    
            print(acc)  
    
#                print(p)
#                #print(labels)
#                print("Step " + str(step) + ", Minibatch Loss= " + \
#                  "{:.4f}".format(loss) + ", Training Accuracy= " + \
#                  "{:.3f}".format(acc) + ", Step Time =" + \
#                  "{:.1f}".format(time.time() - t1))
#                saver.save(sess, save_path='./fnc1_trained_params.ckpt')
    else:
        raise Exception('Choose a mode - "train" to train the model, "pred" to predict the test data')        

print('Program Finished. Total Elapsed Time: {}'.format(time.time() -t00))

#TODO - persist model params for reuse
#TODO - output final classifications for scoring (in excel)
