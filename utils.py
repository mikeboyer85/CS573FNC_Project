###########################################################################
# Author: Cisco Systems, Inc.
# Change Log:
# Date          Author      Desc
# 2017/11/01    Mike Boyer  Ported to python3 and removed theano refs
# 2017/11/05    Mike Boyer  Removed unused split() function and rewrote
#                           "News" class as fnc1_Data() class.
# 2017/11/16    Mike Boyer  Added vecs to fnc1_Data object
###########################################################################

import csv
import _pickle as pickle
import gzip
from time import time
import numpy as np
from collections import defaultdict
from Vectors import *



chars = set([chr(i) for i in range(32,128)])
#character whitelist
stances = {'agree':0,'disagree':1,'discuss':2}
#set up some values for later


def transform(text):
    #convert a string into a np array of approved character indices, starting at 0
    return np.array([ord(i)-32 for i in text if i in chars])

def pad_char(text, padc=-1):
    #take a set of variable length arrays and convert to a matrix with specified fill value
    maxlen = max([len(i) for i in text])
    tmp = np.ones((len(text), maxlen),dtype='int32')
    tmp.fill(padc)
    for i in range(len(text)):
        data = text[i]
        tmp[i,:len(data)]=data
    return tmp

def proc_bodies(fn):
    #process the bodies csv into arrays
    tmp = {}
    with open(fn,'r') as f:
        reader = csv.reader(f)
        next(reader)
        for line in reader:
            bid, text = line
            tmp[bid]=text
    return tmp

class fnc1_Data(object):
    def __init__(self, stancesFile=None, bodiesFile=None, vecs=None):
        
        if stancesFile == None or bodiesFile == None:
            raise ValueError('Set the path to the stances bodies!!!!')
            
        #process files into arrays, etc
        self.bodies = proc_bodies(bodiesFile)
        self.stances = []
        self.body_vecs = {}
        self.stance_vecs = []
        self.vecs = vecs #save the vecs for use in other methods
        
        #Read stances
        with open(stancesFile,'r') as f:
            reader = csv.reader(f)
            next(reader)
            for line in reader:
                if len(line)==2:
                    hl, bid = line
                    stance = None
                else:
                    hl, bid, stance = line
                self.stances.append((hl,bid,stance))

        self.n_stances = len(self.stances)
        
        #Transform the bodies into their vector representation
        for bID, body in self.bodies.items():
            self.body_vecs[bID] = vecs.transform([body])
        
        #Transform the stance (headline only) into its vector representation
        for stance in self.stances:
            self.stance_vecs.append(vecs.transform([stance[0]]))
        
        
    def get_one(self, ridx=None):
        #select a single sample either randomly or by index
        if ridx is None:
            ridx = np.random.randint(0,self.n_stances)
        head = self.stances[ridx]
        body = self.bodies[head[1]]


        return head, body


    def sample(self, n=16, ridx=None):
        #select a batch of samples either randomly or by index
        heads = []
        bodies = []
        stances_d = []
        if ridx is not None:
            for r in ridx:
                head, body_text = self.get_one(r)
                head_text, _, stance = head
                heads.append(head_text)
                bodies.append(body_text)
                stances_d.append(stances[stance])
        else:
            for i in range(n):
                head, body_text = self.get_one()
                head_text, _, stance = head
                heads.append(head_text)
                bodies.append(body_text)
                stances_d.append(stances[stance])


        heads = self.vecs.transform(heads)
        bodies = self.vecs.transform(bodies)
        stances_d = np.asarray(stances_d, dtype='int32')
        #clean up everything and return it

        return heads, bodies, stances_d


    def validate(self):
        #iterate over the dataset in order
        for i in xrange(len(self.headlines)):
            yield self.sample(ridx=[i])
    

if __name__ == '__main__':
    v = GoogleVec()
    v.load()
    val_news = fnc1_Data(stancesFile='./data/train_stances_related_only.csv', vecs=v, bodiesFile='./data/train_bodies.csv')
 

 #   Copyright 2017 Cisco Systems, Inc.
 #
 #   Licensed under the Apache License, Version 2.0 (the "License");
 #   you may not use this file except in compliance with the License.
 #   You may obtain a copy of the License at
 #
 #     http://www.apache.org/licenses/LICENSE-2.0
 #
 #   Unless required by applicable law or agreed to in writing, software
 #   distributed under the License is distributed on an "AS IS" BASIS,
 #   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 #   See the License for the specific language governing permissions and
 #   limitations under the License.
