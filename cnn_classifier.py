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


#The CNN will have 4 layers. hidden size increase with depth (ratios copied from SOLAT team)
weights = {
    'h1': tf.Variable(tf.random_normal([n_embDims, n_hidden])),
    'h2': tf.Variable(tf.random_normal([n_hidden, n_hidden])),
    'h3': tf.Variable(tf.random_normal([n_hidden, n_hidden*2])),
    'h4': tf.Variable(tf.random_normal([n_hidden*2, n_hidden*3])), 
    'out': tf.Variable(tf.random_normal([n_hidden*3, n_classes]))
    }

biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden])),
    'b2': tf.Variable(tf.random_normal([n_hidden])),
    'b3': tf.Variable(tf.random_normal([n_hidden*2])),
    'b4': tf.Variable(tf.random_normal([n_hidden*3])),
    'out': tf.Variable(tf.random_normal([n_classes]))
    }

AdamParams = {
     'learning_rate': 0.001,
     'beta1': 0.1,
     'beta2': 0.001,
     'e': 1e-8
     } 

class fnc1_cnn:
    
    def __init__(self):
        #save params / configuration
        self.theBeginning = datetime.now()
        print('Program start: {:%H:%M:%S}'.format(self.theBeginning))
        
        #Load Google vectorizer. We'll use it a lot from here on out.
        self.gv = GoogleVec()
        self.gv.load()
        print('GoogleVec Loading complete: {:%H:%M:%S} ... Total Elapsed Time: {}'.format(datetime.now(), datetime.now() - self.theBeginning))
    
    def train(self, trainData):
        ############################################
        # Assumptions:
        # 1. Label is always the last column in the stancesFileName
        ############################################
        self.trainStart = datetime.now()
        print('Training Start: {:%H:%M:%S} ... Total Elapsed Time:{}'.format(self.trainStart ,(self.trainStart - self.theBeginning)))
        
        heads, bodies, stances = trainData.sample(n = trainData.n_stances)
        self.feature_vec = np.concatenate((heads, bodies),axis=1)   #The final feature vector for our network is the joined head/body features
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
        
    def getNextTrainData(self, dataSet, labels, nRecords):
        try:
            start = self.trainWindowStart
        except AttributeError:
            start = 0
        
        self.trainWindowStart = start
        
        if self.trainWindowStart == len(dataSet):
            self.trainWindowStart = 0
        elif self.trainWindowStart + nRecords < len(dataSet):
            self.trainWindowEnd = self.trainWindowStart + nRecords
        else:
            self.trainWindowEnd = len(dataSet)
            
        print('returning records {} through {}'.format(self.trainWindowStart, self.trainWindowEnd))        
        ds = dataSet[self.trainWindowStart: self.trainWindowEnd]
        ls = labels[self.trainWindowStart: self.trainWindowEnd]
        
        self.trainWindowStart = self.trainWindowEnd
        return ds, ls 
        
        
        
    def predict(self, bodies):
        raise Exception('Predicition is not implemented yet!!!!')
    
    def validate(self, correctStances, predStances):
        raise Exception('validation is not implemented yet!!!!!')
        
    def neural_net(self, x):
        ##########################################
        # Build the Neural Network
        ##########################################
        #Hidden fully connected layer with 256 neurons
        layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
        #Hidden fully connected layer with 256 neurons
        layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
        #Hidden fully connected layer with 512 neurons
        layer_3 = tf.add(tf.matmul(layer_2, weights['h3']), biases['b3'])
        #Hidden fully connected layer with 768 Neurons
        layer_4 = tf.add(tf.matmul(layer_3, weights['h4']), biases['b4'])
        #Output fully connected layer with a neuron for each class
        out_layer = tf.matmul(layer_4, weights['out']) + biases['out']
        return out_layer

    
if __name__ == '__main__':
    #Step 1 - Read the data
    cnn = fnc1_cnn()   
    #Build the input data from CSV files
    trainDataRead = fnc1_Data(stancesFile = trainStancesFile, bodiesFile = trainBodiesFile, vecs = cnn.gv)
    print('Loaded {} stances'.format(len(trainDataRead.stances)))
    
    #Step 2 - Train the CNN
    cnn.train(trainDataRead)
            
    
    #Step 3 - Generate predictions for test data  
    
    #Step 4 - Generate confusion matrix
    
    #Step 5 - Generate FNC-1 score
    
    
