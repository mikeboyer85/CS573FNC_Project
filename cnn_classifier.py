###################################################
# FNC-1 Neural Network Implementation
# Mike Boyer 
# 11/6/2017 
#
#
###################################################
import numpy
from Vectors import *
from utils import *

trainStancesFile = './data/train_stances_related_only.csv'
trainBodiesFile = './data/train_bodies.csv'
testStancesFile = './data/test_stances_unlabeled.csv'
testBodiesFile = './data/test_bodies.csv'


if __name__ == '__main__':
    #load the vectorizer. We need to for training
    #and for predictions, so load it now and get it out of the way.    
    gv = GoogleVec()
    gv.load()
    
    #Step 1 - Build the CNN in tensorflow
    
    
    #Step 2 - Train the CNN
    #read the train data
    def train(stancesFile = trainStancesFile, bodiesFile = trainBodiesFile):
        
        
    
    #Step 3 - Generate predictions for test dat
    def test(stancesFile = testStancesFile, bodiesFile = testBodiesFile)
    
    
    #Step 4 - Generate confusion matrix
    
    #Step 5 - Generate FNC-1 score
    
    
