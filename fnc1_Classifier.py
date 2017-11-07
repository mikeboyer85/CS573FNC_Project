#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 29 13:46:12 2017

@author: mike
"""
###################################################
# CS 573 - Data Mining and Analysis
# Project - Fake News Challenge: Stance Detection (FNC-1)
# 
# Based on reviewing confusion matrices from the best scoring
# teams in the official FNC-1 challenge, we've opted to prototype 
# this solution, which takes advantage of the best parts of 
# each of the winners' algorithms.
#
# Our approach is broken into the following steps:
#   1. Train a two class classifier for "Related / Unrelated" classification. 
#      Our algorithm is based on SOLAT_in_the_SWEN's Decision Tree classifier.
#       a. Generate features for each headline / body
#       b. Combine individual featuers into a single feature set
#       c. Train a Decision Tree classifier with the features
#       d. (?) Save the trained model to disk so we don't have to re-train
#           every time we use it.
#   2. Train a a three-class classifier for Agree/Disagree/Discuss classification.
#       Not sure what this one looks like yet. :)
###################################################

