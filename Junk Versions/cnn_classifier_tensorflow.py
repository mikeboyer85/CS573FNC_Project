###################################################
# FNC-1 Neural Network Implementation
# Mike Boyer 
# 2017/11/06 
#
# 2017/11/12    Mike Boyer    simplify / rewrite
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
classes = {'agree':0,'disagree':1,'discuss':2}
n_classes = 3       #agree:0, disagree:1, discuss:2
n_hidden = 256      #256 nodes in the first hidden layer. Other layers are multiples of this
n_embDims = 300     #300 embedded dimensions considered in the NN.
batch_size = 32     #Train on 32 stances at a time
num_steps = 10  #training steps. Solat used 35,000,000 
dropout = 0.5
learning_rate = 0.001


def conv_net (x_tensor, is_training, reuse):
    #Convolution Layer with 300 input dims, 256 output dims, width 3, pad 3, stride 1, and relu activation
    conv1 = tf.layers.conv2d(x_tensor, filters=n_hidden, kernel_size = 3, strides = 1, padding = 'same', activation = tf.nn.relu)
    #conv1 = tf.layers.conv1d(x_tensor, filters=n_hidden, kernel_size=3, strides=1, padding='same', activation = tf.nn.relu)
    
    
    
    
    #Apply dropout
    conv1 = tf.layers.dropout(conv1, rate=0.5, training=is_training)
    #Pooling layer with stride (1,2) and pad (1,2)
    conv1 = tf.layers.max_pooling1d(inputs = conv1, pool_size = 2, strides = 2, padding='same')
    
    #Convolution layer with 256 input dims, 256 output dims, width 3, pad 3, stride 1, and relu activation
    conv2 = tf.layers.conv1d(conv1, filters=n_hidden, kernel_size=3, strides=1, padding='same', activation = tf.nn.relu)
    conv2 = tf.layers.dropout(conv2, rate=0.5, training=is_training)
    conv2 = tf.layers.max_pooling1d(inputs=conv2, pool_size = 2, strides = 2, padding = 'same')
    
    #Convolution layer with 256 input dims, 512 output dims, width 3, pad 3, stride 1, and relu activation
    conv3 = tf.layers.conv1d(conv2, filters=n_hidden*2, kernel_size=3, strides=1, padding='same', activation = tf.nn.relu)
    conv3 = tf.layers.dropout(conv3, rate=0.5, training=is_training)
    conv3 = tf.layers.max_pooling1d(inputs=conv3, pool_size=2, strides=2, padding='same')
    
    #Convolution layer with 512 input dims, 512 output dims, width 3, pad 3, stride 1, and relu activation
    conv4 = tf.layers.conv1d(conv3, filters=n_hidden*2, kernel_size=3, strides=1, padding='same', activation = tf.nn.relu)
    conv4 = tf.layers.dropout(conv4, rate=0.5, training=is_training)
    conv4 = tf.layers.max_pooling1d(inputs=conv4, pool_size=2, strides=2, padding='same')
    
    #Convolution layer with 512 input dims, 768 output dims, width 3, pad 1, stride 1, and relu activation
    conv5 = tf.layers.conv1d(conv4, filters=n_hidden*3, kernel_size=3, strides=1, padding='same', activation = tf.nn.relu)
    conv5 = tf.layers.dropout(conv5, rate=0.5, training=is_training)
    conv5 = tf.layers.max_pooling1d(inputs=conv5, pool_size=2, strides=2, padding='same')

    return conv5

def full_net(h_tensor, b_tensor, is_training):
    #A fully connected multi-layer perceptron. Final layer is softmax
    
    h_conv = conv_net(h_tensor, is_training=is_training, reuse=False)
    b_conv = conv_net(b_tensor, is_training=is_training, reuse=False)
    

    
    d1 = tf.layers.dense(feature_vec, 1024, activation=tf.nn.relu)
    d2 = tf.layers.dense(d1, 1024, activation = tf.nn.relu)
    d3 = tf.layers.dense(d2, 1024, activation = tf.nn.relu)
    out = tf.layers.dense(d3, n_classes, activation = tf.nn.softmax)
    
    return out

# Define the model function (following TF Estimator Template)
def model_fn(features, labels, mode):
    print('entering model_fn')
    # Build the neural network
    h_tensor = emb_mat[features['heads'].eval()]
    b_tensor = emb_mat[features['bodies'].eval()]

    print(h_tensor.shape)
    print(b_tensor.shape)
    # Because Dropout has different behavior at training and prediction time, we
    # need to create 2 distinct computation graphs that still share the same weights.
    logits_train = full_net(h_tensor, b_tensor, is_training=True)
    logits_test = full_net(h_tensor, b_tensor, is_training=False)

    # Predictions
    pred_classes = tf.argmax(logits_test, axis=1)
    pred_probas = tf.nn.softmax(logits_test)

    # If prediction mode, early return
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode, predictions=pred_classes)

        # Define loss and optimizer
    loss_op = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=logits_train, labels=tf.cast(labels, dtype=tf.int32)))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(loss_op,
                                  global_step=tf.train.get_global_step())

    # Evaluate the accuracy of the model
    acc_op = tf.metrics.accuracy(labels=labels, predictions=pred_classes)

    # TF Estimators requires to return a EstimatorSpec, that specify
    # the different ops for training, evaluating, ...
    estim_specs = tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=pred_classes,
        loss=loss_op,
        train_op=train_op,
        eval_metric_ops={'accuracy': acc_op})

    return estim_specs

##########################################
# Step 1 - Generate features for the data
##########################################
theBeginning = datetime.now()
print('Program start: {:%H:%M:%S}'.format(theBeginning))
v = GoogleVec()
v.load()
emb_mat = v.model.syn0

# Read training data and generate feature matrix
trainData = fnc1_Data(stancesFile = trainStancesFile, bodiesFile = trainBodiesFile, vecs=v)
trainlabels = np.array(trainData.labels)
# Build the Estimator
model = tf.estimator.Estimator(model_fn)

# Define the input function for training
input_fn = tf.estimator.inputs.numpy_input_fn(
    x={'heads':trainData.stances}, y=trainlabels,
    batch_size=batch_size, num_epochs=None, shuffle=True)
# Train the Model
model.train(input_fn, steps=num_steps)

testData = fnc1_Data(stancesFile = testStancesFile, bodiesFile = testBodiesFile, vecs=v)
testHeads, testBodies, testStances = testData.sample(n=testData.n_stances)
# Evaluate the Model
# Define the input function for evaluating
#input_fn = tf.estimator.inputs.numpy_input_fn(
#    x={'heads': testHeads, 'bodies': testBodies}, y=testStances,
#    batch_size=batch_size, shuffle=False)
## Use the Estimator 'evaluate' method
#e = model.evaluate(input_fn)

print("Testing Accuracy:", e['accuracy'])