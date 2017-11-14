###################################################
# FNC-1 Neural Network Implementation
# Mike Boyer 
# 2017/11/06 
#
# 2017/11/12    Mike Boyer    fnc1_cnn design / framework
###################################################
import numpy
import tensorflow as tf
from Vectors import *
from utils import *
from datetime import datetime

trainStancesFile = './data/train_stances_related_only.csv'
trainBodiesFile = './data/train_bodies.csv'
testStancesFile = './data/test_stances_unlabeled.csv'
testBodiesFile = './data/test_bodies.csv'

 
class fnc1_cnn:
    
    def __init__(self):
        #save params / configuration
        self.theBeginning = datetime.now()
        print('Program start: {}'.format(self.theBeginning))
        
        #Load Google vectorizer. We'll use it a lot from here on out.
        #self.gv = GoogleVec()
        #self.gv.load()
        
        
        
        print('GoogleVec Loading complete: {} ... Elapsed Time: {}'.format(datetime.now(), datetime.now() - self.theBeginning))
        
        
    
    def train(self, stancesFileName, bodiesFileName):
        ############################################
        # Assumptions:
        # 1. Label is always the last column in the stancesFileName
        ############################################
        self.trainStart = datetime.now()
        print('Training Start: {} ... Elapsed Time:{}'.format(self.trainStart ,(self.trainStart - self.theBeginning)))
        #raise Exception('Training is not implemented yet!!!!!')
        
    def predict(self, bodies):
        raise Exception('Predicition is not implemented yet!!!!')
    
    def validate(self, correctStances, predStances):
        raise Exception('validation is not implemented yet!!!!!')
    
if __name__ == '__main__':
    cnn = fnc1_cnn()   
    #Step 1 - Build the CNN in tensorflow
    
    
    #Step 2 - Train the CNN
    cnn.train(bodies=None, stances=None)
            
    
    #Step 3 - Generate predictions for test data  
    
    #Step 4 - Generate confusion matrix
    
    #Step 5 - Generate FNC-1 score
    
    
